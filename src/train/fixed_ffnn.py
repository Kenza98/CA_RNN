"""
Experiment 2: train the feed-forward baseline (FeedForwardNN) alone, with its
initial hand-picked hyperparameters (before any Optuna search), and record
training metrics to illustrate its training behaviour in the thesis.

It trains on the experiment-1 data (1_nca_* or 1_nn_*), so it is directly
comparable with the RNNs trained by train_fixed.py. The FFNN flattens the
(tau, input_dim) window into one vector and has no recurrence.

Outputs, in outputs/exp2_ffnn/<tag>/:
  - train_loss.png              per-step training MSE
  - train_loss_convergence.png  per-step MSE with per-epoch means
  - train_val_loss.png          per-epoch train vs validation MSE
  - grad_norms.png              per-parameter gradient norm at every step
                                (after clipping at max_norm=1.0)
  - tanh_activations.png        hidden activations before vs after training
                                (saturation check for the tanh + Xavier choice)
  - metrics.pt                  the raw numbers behind every plot
The checkpoint goes to models/[<folder-name>/]<tag>.pt.

Usage:
    python -m src.train.fixed_ffnn --dataset nca
"""

import os
from datetime import datetime
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.seed import set_seed
from src.utils.train_loop import train_model
from src.utils.normalize import normalize
from src.utils.plots_model import (
    plot_loss_per_epoch,
    plot_loss_convergence,
    plot_grad_hist,
    plot_train_val_loss,
    plot_activation_hist,
)
from src.models.ffnn import FeedForwardNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs" / "exp2_ffnn"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["nca", "nn"],
    required=True,
    help="Which experiment-1 dataset to train on",
)
parser.add_argument(
    "--use-gpu",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use GPU if CUDA available (--no-use-gpu to force CPU)",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--width", type=int, default=64, help="Hidden layer width")
parser.add_argument("--num-layers", type=int, default=1, help="Number of hidden layers")
parser.add_argument("--num-epochs", type=int, default=10, help="Training epochs")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--folder-name", type=str, help="Subfolder of models/ for the checkpoint")
args = parser.parse_args()

set_seed(args.seed)
if args.folder_name:
    MODEL_DIR = MODEL_DIR / args.folder_name
MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}", flush=True)

job_id = os.environ.get("SLURM_JOB_ID")
timestamp = datetime.now().strftime("%m%d_%H%M")
run_id = f"gpu_{job_id}" if device.type == "cuda" else f"cpu_{timestamp}"

prefix = f"1_{args.dataset}"


def load_split(split):
    return torch.load(DATA_DIR / f"{prefix}_{split}.pt", map_location="cpu", weights_only=False)


# normalise both splits with the *training* statistics
train_data, val_data = load_split("train"), load_split("val")
mean, std = train_data["config"]["global_mean"], train_data["config"]["global_std"]
X, Y = normalize(train_data["X"], train_data["Y"], mean, std)
X_val, Y_val = normalize(val_data["X"], val_data["Y"], mean, std)
del train_data, val_data

train_loader = DataLoader(
    TensorDataset(X, Y), batch_size=args.batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    TensorDataset(X_val, Y_val), batch_size=4096, shuffle=False, num_workers=4
)

input_dim = X.shape[-1]
seq_length = X.shape[1]  # tau, read from the data
output_dim = 1

model = FeedForwardNN(
    input_dim, args.width, output_dim, num_layers=args.num_layers, seq_length=seq_length
)
model_name = model.__class__.__name__
tag = f"{model_name.lower()}_{prefix}_{run_id}"
run_dir = OUT_DIR / tag
run_dir.mkdir(parents=True, exist_ok=True)
n_params = sum(p.numel() for p in model.parameters())
print(
    f"Model: {model_name} | dataset={args.dataset} | width={args.width} | layers={args.num_layers} "
    f"| params={n_params} | epochs={args.num_epochs} | batch={args.batch_size} | lr={args.lr:.2e} "
    f"| seed={args.seed}",
    flush=True,
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += criterion(model(xb), yb).item() * len(xb)
        n += len(xb)
    return total / n


@torch.no_grad()
def tanh_activations(model, x):
    """Outputs of every tanh layer on batch x, as {layer name: flat CPU tensor}."""
    model.eval()
    acts, hooks = {}, []
    tanh_layers = [m for m in model.net if isinstance(m, nn.Tanh)]
    for i, layer in enumerate(tanh_layers):
        name = f"hidden layer {i + 1}"
        hooks.append(layer.register_forward_hook(
            lambda _m, _in, out, name=name: acts.__setitem__(name, out.flatten().cpu())
        ))
    model.to(device)(x.to(device))
    for h in hooks:
        h.remove()
    return acts


probe = X_val[:4096]  # fixed validation batch for the activation histograms
acts_before = tanh_activations(model, probe)

# one epoch at a time so the validation loss can be measured after each epoch
train_loss, grad_history = [], {}
train_epoch_loss, val_epoch_loss = [], []
for epoch in range(args.num_epochs):
    print(f"== Epoch {epoch + 1}/{args.num_epochs} ==", flush=True)
    step_loss, step_grads = train_model(model, train_loader, optimizer, criterion, 1, device)
    train_loss += step_loss
    for name, norms in step_grads.items():
        grad_history.setdefault(name, []).extend(norms)
    train_epoch_loss.append(sum(step_loss) / len(step_loss))
    val_epoch_loss.append(evaluate(model, val_loader))
    print(f"Val loss: {val_epoch_loss[-1]:.8f}", flush=True)

acts_after = tanh_activations(model, probe)

checkpoint = {
    f"{model_name}StateDict": model.state_dict(),
    "model_type": model_name,
    "dataset": args.dataset,
    "hidden_dim": args.width,
    "num_layers": args.num_layers,
    "lr": args.lr,
    "seed": args.seed,
    "seq_length": seq_length,
    "global_mean": mean,
    "global_std": std,
}
model_file = MODEL_DIR / f"{tag}.pt"
torch.save(checkpoint, model_file)
print(f"Saved to {model_file}", flush=True)

torch.save(
    {
        "args": vars(args),
        "n_params": n_params,
        "steps_per_epoch": len(train_loader),
        "train_loss": train_loss,
        "train_epoch_loss": train_epoch_loss,
        "val_epoch_loss": val_epoch_loss,
        "grad_history": grad_history,
        "acts_before": acts_before,
        "acts_after": acts_after,
    },
    run_dir / "metrics.pt",
)

plot_kw = dict(model_name=model_name, dataset=args.dataset)
plot_loss_per_epoch(train_loss, run_dir / "train_loss.png", **plot_kw)
plot_loss_convergence(train_loss, len(train_loader), run_dir / "train_loss_convergence.png", **plot_kw)
plot_train_val_loss(train_epoch_loss, val_epoch_loss, run_dir / "train_val_loss.png", **plot_kw)
plot_grad_hist(grad_history, run_dir / "grad_norms.png")
plot_activation_hist(acts_before, acts_after, run_dir / "tanh_activations.png", **plot_kw)
print(f"Plots and metrics in {run_dir}", flush=True)
