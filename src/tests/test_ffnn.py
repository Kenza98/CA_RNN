"""
Experiment 2: evaluate FeedForwardNN checkpoints from src/train/fixed_ffnn.py
on the held-out experiment-1 test sets (1_nca_test.pt, 1_nn_test.pt).

One-step evaluation, same as src/tests/ablation_1.py for the RNNs: every test
window predicts the next value of its cell, errors are reported in degC.
A persistence baseline (next value = last observed value of the cell) is
computed on the same samples for reference.

The test set is normalised with the train-set mean/std stored in the checkpoint.

Usage:
    python -m src.tests.test_ffnn                          # latest FFNN checkpoint per dataset
    python -m src.tests.test_ffnn --checkpoint models/<file>.pt [...]

Writes results/exp2_ffnn_test_<checkpoint>.json (same layout as ablation_1
JSONs, under the "ffnn" key) and a histogram of absolute errors per checkpoint
in outputs/exp2_ffnn/test/.
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.ffnn import FeedForwardNN
from src.utils.normalize import normalize
from src.utils.evaluate import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOT_DIR = PROJECT_ROOT / "outputs" / "exp2_ffnn" / "test"

# index of the cell's own value in the feature axis (3x3 neighbourhood is row-major)
CENTRE_FEATURE = {"nca": 4, "nn": 0}
QUANTILES = [0.5, 0.9, 0.95, 0.99]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    nargs="+",
    type=Path,
    help="FFNN checkpoint(s) to evaluate; omit for the latest one per dataset under models/",
)
parser.add_argument(
    "--use-gpu",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use GPU if CUDA available (--no-use-gpu to force CPU)",
)
parser.add_argument("--batch-size", type=int, default=4096)
args = parser.parse_args()

device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}", flush=True)


def latest_checkpoints():
    """Most recent feedforwardnn_1_<dataset>_*.pt per dataset, anywhere under models/."""
    found = []
    for dataset in CENTRE_FEATURE:
        matches = sorted(
            MODEL_DIR.rglob(f"feedforwardnn_1_{dataset}_*.pt"), key=lambda f: f.stat().st_mtime
        )
        if matches:
            found.append(matches[-1])
        else:
            print(f"No FFNN checkpoint found for {dataset}, skipping.")
    return found


def error_stats(se, ae, std):
    """MSE/RMSE/MAE and absolute-error quantiles, converted from normalised units to degC."""
    mse = se.mean().item() * std**2
    ae_c = ae * std
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": ae_c.mean().item(),
        "ae_quantiles": {
            str(q): v.item() for q, v in zip(QUANTILES, torch.quantile(ae_c, torch.tensor(QUANTILES)))
        },
    }


def print_stats(name, s):
    qs = " | ".join(f"q{float(q):.2f}={v:.4f}" for q, v in s["ae_quantiles"].items())
    print(f"{name:<12} MSE={s['mse']:.6f} °C² | RMSE={s['rmse']:.6f} °C | MAE={s['mae']:.6f} °C")
    print(f"{'':<12} |error| quantiles (°C): {qs}")


checkpoints = args.checkpoint or latest_checkpoints()
if not checkpoints:
    raise SystemExit("Nothing to evaluate.")
RESULTS_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

for ckpt_path in checkpoints:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("model_type") != "FeedForwardNN":
        print(f"{ckpt_path} is a {ckpt.get('model_type')} checkpoint, not an FFNN, skipping.")
        continue
    dataset = ckpt["dataset"]
    mean, std = float(ckpt["global_mean"]), float(ckpt["global_std"])
    print(f"\n=== {ckpt_path.name} (dataset={dataset}) ===", flush=True)

    test = torch.load(DATA_DIR / f"1_{dataset}_test.pt", map_location="cpu", weights_only=False)
    X, Y = normalize(test["X"], test["Y"], mean, std)
    if X.shape[1] != ckpt["seq_length"]:
        raise ValueError(f"test windows have tau={X.shape[1]}, model was trained with {ckpt['seq_length']}")

    model = FeedForwardNN(
        X.shape[-1], ckpt["hidden_dim"], 1,
        num_layers=ckpt["num_layers"], seq_length=ckpt["seq_length"],
    )
    model.load_state_dict(ckpt["FeedForwardNNStateDict"])

    loader = DataLoader(TensorDataset(X, Y), batch_size=args.batch_size, shuffle=False)
    out = evaluate_model(model, loader, device)  # normalised units
    model_stats = error_stats(out["squared_error"], out["absolute_error"], std)

    # persistence baseline: the cell's last observed value
    persist = X[:, -1, CENTRE_FEATURE[dataset]] - Y.squeeze(-1)
    persist_stats = error_stats(persist**2, persist.abs(), std)

    print(f"{len(Y)} test samples")
    print_stats("FFNN", model_stats)
    print_stats("persistence", persist_stats)
    print(f"FFNN RMSE / persistence RMSE = {model_stats['rmse'] / persist_stats['rmse']:.3f}")

    hyperparams = {k: ckpt[k] for k in ("hidden_dim", "num_layers", "lr", "seed", "seq_length")}
    result = {
        "checkpoint": str(ckpt_path),
        "n_samples": len(Y),
        "test results": {
            dataset: {
                "ffnn": {"hyperparams": hyperparams, **model_stats},
                "persistence": persist_stats,
            }
        },
    }
    json_file = RESULTS_DIR / f"exp2_ffnn_test_{ckpt_path.stem}.json"
    with open(json_file, "w") as f:
        json.dump(result, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 6))
    upper = float(torch.quantile(persist.abs() * std, 0.99))
    for label, ae in (("FFNN", out["absolute_error"] * std), ("persistence", persist.abs() * std)):
        ax.hist(ae.numpy(), bins=100, range=(0, upper), alpha=0.5, density=True,
                label=f"{label} (MAE {ae.mean().item():.3f} °C)")
    ax.set_xlabel("Absolute error (°C)")
    ax.set_ylabel("Density")
    ax.set_title(f"FeedForwardNN on {dataset} — test-set absolute errors")
    ax.legend()
    fig.tight_layout()
    plot_file = PLOT_DIR / f"{ckpt_path.stem}_abs_error_hist.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {json_file}\nSaved {plot_file}", flush=True)
