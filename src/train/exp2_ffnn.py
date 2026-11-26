"""
Experiment 2: Optuna hyperparameter search for the FeedForwardNN baseline.

Same setup as the experiment-1 searches (exp1_{gru,lstm,rnn}.py): train on
<data-tag>_<dataset>_train.pt, score on the validation split, val MSE in degC²
is the objective, trials stored in optuna/<study>.db (study = ffnn_<data-tag>_<dataset>),
so src/tests/optuna_report.py --model ffnn works on it.

Differences with experiment 1:
  - the learning rate is searched too (the FFNN turned out to be sensitive to it),
  - validation MSE is reported after every epoch, so bad trials are pruned early,
  - every trial is logged to TensorBoard under runs/exp2/<study>/trial_XXXX
    (per-step loss and gradient norm, per-epoch train/val loss, hyperparameters),
    and the study-level Optuna plots go to runs/exp2/<study>/study.

Usage:
    python -m src.train.exp2_ffnn --dataset nca --n-trials 60
    tensorboard --logdir runs/exp2
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from src.models.ffnn import FeedForwardNN
from src.utils.normalize import normalize
from src.utils.seed import set_seed
from src.utils.train_loop import train_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OPTUNA_DIR = PROJECT_ROOT / "optuna"
RESULTS_DIR = PROJECT_ROOT / "results"
TB_DIR = PROJECT_ROOT / "runs" / "exp2"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["nca", "nn"], required=True)
parser.add_argument(
    "--data-tag",
    default="1",
    help="Prefix of the data files, <data-tag>_<dataset>_<split>.pt. '1' = experiment-1 data "
    "(3 years of training data, tau=6); change it once the longer datasets exist",
)
parser.add_argument("--n-trials", type=int, default=45, help="Number of new Optuna trials")
parser.add_argument("--num-epochs", type=int, default=5, help="Epochs per trial (exp1 used 5)")
parser.add_argument("--seed", type=int, default=42, help="Seed for the sampler and every trial")
parser.add_argument(
    "--use-gpu",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use GPU if CUDA available (--no-use-gpu to force CPU)",
)
parser.add_argument(
    "--study-name", default=None, help="Override the study name (default ffnn_<data-tag>_<dataset>)"
)
parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
parser.add_argument(
    "--grad-log-every", type=int, default=10, help="Log the gradient norm to TensorBoard every N steps"
)
args = parser.parse_args()

device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}", flush=True)

prefix = f"{args.data_tag}_{args.dataset}"
study_name = args.study_name or f"ffnn_{prefix}"
job_id = os.environ.get("SLURM_JOB_ID")
run_id = f"job_{job_id}" if job_id else f"local_{datetime.now().strftime('%m%d_%H%M')}"
OPTUNA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
study_tb_dir = TB_DIR / study_name


def load_split(split):
    return torch.load(DATA_DIR / f"{prefix}_{split}.pt", map_location="cpu", weights_only=False)


# both splits normalised with the training statistics
train_data, val_data = load_split("train"), load_split("val")
mean, std = train_data["config"]["global_mean"], train_data["config"]["global_std"]
X, Y = normalize(train_data["X"], train_data["Y"], mean, std)
X_val, Y_val = normalize(val_data["X"], val_data["Y"], mean, std)
del train_data, val_data

input_dim = X.shape[-1]
seq_length = X.shape[1]
print(f"Data {prefix}: train {tuple(X.shape)}, val {tuple(X_val.shape)}, tau={seq_length}", flush=True)

train_dataset = TensorDataset(X, Y)
val_loader = DataLoader(
    TensorDataset(X_val, Y_val), batch_size=4096, shuffle=False,
    num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
)
_train_loaders = {}


def get_train_loader(batch_size):
    """One loader per batch size, so worker processes aren't respawned every trial."""
    if batch_size not in _train_loaders:
        _train_loaders[batch_size] = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        )
    return _train_loaders[batch_size]


criterion = nn.MSELoss()


@torch.no_grad()
def val_mse_mae(model):
    """Validation MSE (degC²) and MAE (degC)."""
    model.eval()
    se, ae, n = 0.0, 0.0, 0
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        err = model(xb) - yb
        se += (err**2).sum().item()
        ae += err.abs().sum().item()
        n += len(yb)
    return se / n * std**2, ae / n * std


def objective(trial):
    width = trial.suggest_categorical("width", [16, 32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    set_seed(args.seed)  # same init and batch order for every trial: only the hyperparameters differ
    train_loader = get_train_loader(batch_size)
    model = FeedForwardNN(input_dim, width, 1, num_layers=num_layers, seq_length=seq_length)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trial.set_user_attr("n_params", sum(p.numel() for p in model.parameters()))

    print(
        f"\nTrial {trial.number} | width={width} | layers={num_layers} | bs={batch_size} | lr={lr:.2e}",
        flush=True,
    )
    writer = SummaryWriter(study_tb_dir / f"trial_{trial.number:04d}")
    step = 0
    val_mse, val_mae, best_val, best_epoch = None, None, float("inf"), 0
    try:
        for epoch in range(1, args.num_epochs + 1):
            step_loss, grad_hist = train_model(model, train_loader, optimizer, criterion, 1, device)

            # per-step training loss (degC²) and total gradient norm (after clipping)
            for i, loss in enumerate(step_loss):
                writer.add_scalar("loss/train_step", loss * std**2, step + i)
            names = list(grad_hist)
            for i in range(0, len(step_loss), args.grad_log_every):
                total = sum(grad_hist[n][i] ** 2 for n in names) ** 0.5
                writer.add_scalar("grad_norm/total", total, step + i)
            step += len(step_loss)

            train_mse = sum(step_loss) / len(step_loss) * std**2
            val_mse, val_mae = val_mse_mae(model)
            writer.add_scalars("loss/epoch", {"train": train_mse, "val": val_mse}, epoch)
            writer.add_scalar("val/mae", val_mae, epoch)
            print(f"  epoch {epoch}: train MSE={train_mse:.6f} °C² | val MSE={val_mse:.6f} °C²", flush=True)

            if val_mse < best_val:
                best_val, best_epoch = val_mse, epoch
            trial.report(val_mse, epoch)
            if trial.should_prune():
                print(f"  pruned at epoch {epoch}", flush=True)
                raise optuna.TrialPruned()
    finally:
        writer.add_hparams(
            {"width": width, "num_layers": num_layers, "batch_size": batch_size, "lr": lr},
            {"hparam/val_mse": val_mse if val_mse is not None else float("nan"),
             "hparam/best_val_mse": best_val},
            run_name=".",
        )
        writer.close()

    trial.set_user_attr("train_mse", train_mse)
    trial.set_user_attr("val_mae", val_mae)
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_val_mse", best_val)
    print(f"----> val MSE={val_mse:.6f} °C² | val MAE={val_mae:.6f} °C | best epoch={best_epoch}", flush=True)
    return val_mse  # last epoch, the one a checkpoint trained with these settings would have


study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=args.seed),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    storage=f"sqlite:///{OPTUNA_DIR / study_name}.db",
    study_name=study_name,
    load_if_exists=True,
)
study.set_user_attr("num_epochs", args.num_epochs)
study.set_user_attr("seq_length", seq_length)
study.optimize(objective, n_trials=args.n_trials)


# PED-ANOVA: no sklearn needed (the default fANOVA evaluator requires it)
IMPORTANCE = optuna.importance.PedAnovaImportanceEvaluator()


def log_study_plots():
    """Optuna's summary plots, as images in TensorBoard (tab IMAGES, run 'study')."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as ovm

    plots = {
        "optimization_history": ovm.plot_optimization_history,
        "param_importances": lambda st: ovm.plot_param_importances(st, evaluator=IMPORTANCE),
        "slice": ovm.plot_slice,
        "parallel_coordinate": ovm.plot_parallel_coordinate,
        "intermediate_values": ovm.plot_intermediate_values,
    }
    writer = SummaryWriter(study_tb_dir / "study")
    for name, fn in plots.items():
        try:
            ax = fn(study)
            fig = (ax.flat[0] if hasattr(ax, "flat") else ax).figure
            fig.set_size_inches(12, 7)
            writer.add_figure(f"optuna/{name}", fig, global_step=len(study.trials))
            plt.close(fig)
        except Exception as e:  # e.g. importances need >1 completed trial
            print(f"Skipped plot {name}: {e}", flush=True)
    writer.close()


log_study_plots()

states = [t.state for t in study.trials]
n_done = states.count(optuna.trial.TrialState.COMPLETE)
n_pruned = states.count(optuna.trial.TrialState.PRUNED)
best = study.best_trial
summary = {
    "study_name": study_name,
    "run_id": run_id,
    "n_trials": len(study.trials),
    "n_complete": n_done,
    "n_pruned": n_pruned,
    "num_epochs": args.num_epochs,
    "seq_length": seq_length,
    "best_trial": best.number,
    "best_val_mse": best.value,
    "best_params": best.params,
    "best_user_attrs": best.user_attrs,
}
try:
    summary["param_importances"] = optuna.importance.get_param_importances(study, evaluator=IMPORTANCE)
except Exception as e:
    print(f"Skipped param importances: {e}", flush=True)
out_file = RESULTS_DIR / f"exp2_optuna_{study_name}.json"
with open(out_file, "w") as f:
    json.dump(summary, f, indent=2)

p = best.params
print("\n=== Optuna Search Complete ===")
print(f"Trials: {len(study.trials)} ({n_done} complete, {n_pruned} pruned)")
print(f"Best trial {best.number}: val MSE={best.value:.6f} °C² | params={p}")
print(f"Summary: {out_file}")
print(
    "Train the final checkpoint with:\n"
    f"  python -m src.train.fixed_ffnn --dataset {args.dataset} --width {p['width']} "
    f"--num-layers {p['num_layers']} --batch-size {p['batch_size']} --lr {p['lr']:.3g} "
    f"--num-epochs {args.num_epochs}"
)
