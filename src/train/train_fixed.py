"""
Train a single RNN/GRU/LSTM model with fixed hyperparameters on experiment 1
data (1_nca_* or 1_nn_*), no Optuna search.

The optuna search in exp1_{gru,lstm,rnn}.py showed <0.0002 val MSE spread
across configs, and that more epochs (past 3) made results worse.
so small networks trained briefly are what matters, not the exact config.

Evaluation on the held-out test set happens separately in
src/tests/ablation_1.py, using the checkpoint saved here.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.seed import set_seed
from src.utils.train_loop import train_model
from src.utils.normalize import normalize
from src.utils.plots_model import plot_loss_per_epoch, plot_grad_hist
from src.models.gru import GRU
from src.models.lstm import LSTM
from src.models.VanillaRNN import VanillaRNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

MODEL_CLASSES = {"gru": GRU, "lstm": LSTM, "rnn": VanillaRNN}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", choices=list(MODEL_CLASSES), required=True, help="Which model to train"
)
parser.add_argument(
    "--dataset",
    choices=["nca", "nn"],
    required=True,
    help="Which experiment-1 dataset to train on",
)
parser.add_argument("--use-gpu", action="store_true", help="Use GPU if CUDA available")
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)
parser.add_argument("--hidden-dim", type=int, default=28, help="Hidden state size")
parser.add_argument(
    "--num-layers", type=int, default=5, help="Number of stacked recurrent layers"
)
parser.add_argument("--num-epochs", type=int, default=3, help="Training epochs")
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()

set_seed(args.seed)

device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}", flush=True)

job_id = os.environ.get("SLURM_JOB_ID")
timestamp = datetime.now().strftime("%m%d_%H%M")
run_id = f"gpu_{job_id}" if device.type == "cuda" else f"cpu_{timestamp}"

prefix = f"1_{args.dataset}"

data = torch.load(
    DATA_DIR / f"{prefix}_train.pt", map_location="cpu", weights_only=False
)
X, Y = data["X"], data["Y"]

mean = data["config"]["global_mean"]
std = data["config"]["global_std"]
X, Y = normalize(X, Y, mean, std)

train_loader = DataLoader(
    TensorDataset(X, Y), batch_size=args.batch_size, shuffle=True, num_workers=4
)

input_dim = X.shape[-1]
output_dim = 1

model_class = MODEL_CLASSES[args.model]
model = model_class(input_dim, args.hidden_dim, output_dim, num_layers=args.num_layers)
model_name = model.__class__.__name__
tag = f"{model_name.lower()}_{prefix}_{run_id}"
print(
    f"Model: {model_name} | dataset={args.dataset} | hidden={args.hidden_dim} | layers={args.num_layers} "
    f"| epochs={args.num_epochs} | lr={args.lr:.2e} | seed={args.seed}",
    flush=True,
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

train_loss, grad_history = train_model(
    model, train_loader, optimizer, criterion, args.num_epochs, device
)

checkpoint = {
    f"{model_name}StateDict": model.state_dict(),
    "model_type": model_name,
    "dataset": args.dataset,
    "hidden_dim": args.hidden_dim,
    "num_layers": args.num_layers,
    "lr": args.lr,
    "seed": args.seed,
    "global_mean": mean,
    "global_std": std,
}

model_file = MODEL_DIR / f"{tag}.pt"
torch.save(checkpoint, model_file)
print(f"Saved to {model_file}", flush=True)

plot_loss_per_epoch(train_loss, OUT_DIR / f"{tag}_train_loss.png")
plot_grad_hist(grad_history, OUT_DIR / f"{tag}_grad.png")
