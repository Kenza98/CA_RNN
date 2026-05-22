import time, os
import argparse
from datetime import datetime
from pathlib import Path
import optuna
from src.utils.plots_model import *
from src.utils.train_loop import train_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.VanillaRNN import *

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

parser = argparse.ArgumentParser()
parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
args = parser.parse_args()

device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}", flush=True)

job_id = os.environ.get("SLURM_JOB_ID")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = f"gpu_{job_id}" if device.type == "cuda" else f"cpu_{timestamp}"

# load data once, reuse across trials
load_file = DATA_DIR / "sst_train_set_norm.pt"
data = torch.load(load_file, map_location="cpu", weights_only=False)
X = data["X"]
Y = data["Y"]

# load val set for evaluation
val_file = DATA_DIR / "sst_val_set_norm.pt" 
val_data = torch.load(val_file, map_location="cpu", weights_only=False)
X_val = val_data["X"]
Y_val = val_data["Y"]

input_dim = 9
output_dim = 1
lr = 1e-4

#mean = data["mean"]
#std = data["std"]

def objective(trial):
    # search space
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 56, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    #lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    num_epochs = 20  # keep low for search — full training comes after

    train_dataset = TensorDataset(X, Y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = VanillaRNN(input_dim, hidden_dim, output_dim, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_model(model, train_loader, optimizer, criterion, num_epochs, device)

    # evaluate on val set
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, Y_batch)
            val_losses.append(loss.item())

    val_mse = sum(val_losses) / len(val_losses)
    print(f"Trial {trial.number} | hidden={hidden_dim} layers={num_layers} lr={lr:.2e} bs={batch_size} -> val_MSE={val_mse:.6f}", flush=True)
    return val_mse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=args.n_trials)

print("\n=== Optuna Search Complete ===")
print(f"Best val MSE: {study.best_value:.6f}")
print(f"Best params: {study.best_params}")

# save study results
import pandas as pd
df = study.trials_dataframe()
df.to_csv(OUT_DIR / f"optuna_results_{run_id}.csv", index=False)
print(f"Saved trial results to {OUT_DIR}/optuna_results_{run_id}.csv")
