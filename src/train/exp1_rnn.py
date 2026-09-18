"""
Train a Vanilla RNN model on experiment 1 data (1_nca_* or 1_nn_*) using Optuna.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import argparse
import optuna
import torch.nn as nn
import torch.optim as optim
import os
import csv
from src.utils.train_loop import train_model
from src.utils.evaluate import evaluate_model
from src.utils.seed import set_seed
from datetime import datetime
from src.models.VanillaRNN import VanillaRNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["nca", "nn"],
    required=True,
    help="Which experiment-1 dataset to train on",
)
parser.add_argument(
    "--use-gpu", action="store_true", help="Use GPU if CUDA module available"
)
parser.add_argument("--n-trials", type=int, default=45, help="Number of Optuna trials")
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)
args = parser.parse_args()

set_seed(args.seed)

device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}", flush=True)

job_id = os.environ.get("SLURM_JOB_ID")
timestamp = datetime.now().strftime("%m%d_%H%M")
run_id = f"gpu_{job_id}" if device.type == "cuda" else f"cpu_{timestamp}"

prefix = f"1_{args.dataset}"

# load data once
data = torch.load(
    DATA_DIR / f"{prefix}_train.pt", map_location="cpu", weights_only=False
)
X, Y = data["X"], data["Y"]

# normalize using training set statistics
mean = data["config"]["global_mean"]
std = data["config"]["global_std"]
# standardize and normalize by hand
X = (X - mean) / std
Y = (Y - mean) / std

# validation set
val_data = torch.load(
    DATA_DIR / f"{prefix}_val.pt", map_location="cpu", weights_only=False
)
X_val, Y_val = val_data["X"], val_data["Y"]
X_val = (X_val - mean) / std
Y_val = (Y_val - mean) / std

input_dim = X.shape[-1]
output_dim = 1
lr = 1e-4
num_epochs = 5

train_dataset = TensorDataset(X, Y)
val_dataset = TensorDataset(X_val, Y_val)
_loaders = {}


def get_loaders(batch_size):
    """Cache loaders per batch_size so worker processes aren't respawned every trial."""
    if batch_size not in _loaders:
        _loaders[batch_size] = (
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
            ),
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
            ),
        )
    return _loaders[batch_size]


def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    train_loader, val_loader = get_loaders(batch_size)

    model = VanillaRNN(input_dim, hidden_dim, output_dim, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_model(model, train_loader, optimizer, criterion, num_epochs, device)

    # errors are y_hat - y, so the shared mean cancels out on destandardize:
    # only the std scaling remains (MSE scales by std**2, MAE by std)
    train_metrics = evaluate_model(model, train_loader, device)
    val_metrics = evaluate_model(model, val_loader, device)
    train_loss = train_metrics["mse"].item() * std**2
    val_mse = val_metrics["mse"].item() * std**2
    val_mae = val_metrics["mae"].item() * std

    trial.set_user_attr("train_loss", train_loss)
    trial.set_user_attr("val_mae", val_mae)

    print(
        f"Trial {trial.number} | hidden={hidden_dim} | layers={num_layers} | bs={batch_size} | epochs={num_epochs}",
        flush=True,
    )
    print(
        f"----> train_loss={train_loss:.6f} °C² | val_MSE={val_mse:.6f} °C² | val_MAE={val_mae:.6f} °C",
        flush=True,
    )
    return val_mse


fieldnames = [
    "model",
    "hidden_dim",
    "num_layers",
    "batch_size",
    "num_epochs",
    "train_loss",
    "val_mse",
    "val_mae",
]
out_file = OUT_DIR / f"optuna_rnn_{prefix}_results_{run_id}.csv"
with open(out_file, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def log_trial(study, trial):
    """Append this trial's row as soon as it completes, so results survive a crash/timeout."""
    if trial.value is None:
        return
    with open(out_file, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(
            {
                "model": "rnn",
                "num_epochs": num_epochs,
                **trial.params,
                "train_loss": trial.user_attrs.get("train_loss"),
                "val_mse": trial.value,
                "val_mae": trial.user_attrs.get("val_mae"),
            }
        )


study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed)
)
study.optimize(objective, n_trials=args.n_trials, callbacks=[log_trial])

print("\n=== Optuna Search Complete ===")
print(f"Best val MSE: {study.best_value:.6f} °C²")
print(f"Best params: {study.best_params}")
print(f"Saved trial results to {out_file}")
