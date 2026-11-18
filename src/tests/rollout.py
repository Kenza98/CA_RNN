"""Autoregressive rollout of trained models over the full SST grid.

Starts from real SST frames, predicts the next grid, appends it to the input and repeats for ``n_steps``.
The model sees no real frame after the seed, except where a fill function decides so.

Building blocks:
- ``predict_next_grid``: one step from a window of ``seq_length`` grids, for RNN + CA (3x3 neighborhood) or RNN without CA (cell's own history).
  Returns NaN on cells the model cannot predict (land, exterior ring, coastal cells touching land).
- Fill functions: give those unpredictable sea cells a value before the grid is fed back.
  Common signature: ``fill(pred, predicted_mask, sea_mask, window, true_next) -> grid``.
    * ``fill_with_ground_truth``: real values (leaks truth into the rollout).
    * ``fill_with_neighbour_mean``: mean of the available neighbours in the predicted grid (no truth).
- Rollout functions, one per framing:
    * ``rollout_ca_rnn``: RNN + CA, seeded with a real window.
    * ``rollout_neighbour_average``: baseline, each step is the 3x3 mean of the previous grid (no model).
    * \\TODO ``rollout_rnn`` (no CA), ``rollout_ffnn``.

Errors are computed on model-predicted cells only, so filled cells never count.

Usage:
    python -m src.tests.rollout --checkpoint models/lstm_1_nca_gpu_102059.pt
    python -m src.tests.rollout --checkpoint ... --fill neighbour_mean --start 30
"""


import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import xarray as xr

from src.data.download import raw_path
from src.data.extractors import extract_neighborhood, neighborhood_valid_mask
from src.models.gru import GRU
from src.models.lstm import LSTM
from src.models.VanillaRNN import VanillaRNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "ca_results"

SEQ_LENGTH = 6
MODEL_CLASSES = {"GRU": GRU, "LSTM": LSTM, "VanillaRNN": VanillaRNN}
INPUT_DIMS = {"nca": 9, "nn": 1}  # matches the extractors in src/data/extractors.py


# STEP 0: Load raw data, needed for computing outside ring + for computing errors

def load_sst_series(split="test"):
    """Load the raw SST time series of a split as a float tensor (T, H, W), in degC."""
    with xr.open_dataset(raw_path(split)) as ds:
        sst = ds["thetao"]
        if "depth" in sst.dims:
            sst = sst.isel(depth=0)
        return torch.from_numpy(sst.values).float()


def load_checkpoint(path, device):
    """Rebuild a previously trained model saved in a checkpoint.
    Returns (model, mean, std, meta), where mean/std are the train-set stats the model was trained with and meta holds the checkpoint's hyperparameters.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_type = ckpt["model_type"]
    model = MODEL_CLASSES[model_type](
        INPUT_DIMS[ckpt["dataset"]],
        ckpt["hidden_dim"],
        1,
        num_layers=ckpt["num_layers"],
    )
    model.load_state_dict(ckpt[f"{model_type}StateDict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if not k.endswith("StateDict")}
    return model, float(ckpt["global_mean"]), float(ckpt["global_std"]), meta


# STEP 1: One CA step == window ---> next grid

#decorator wrapping subsequent function into a with ... : statement.
@torch.no_grad()
def predict_next_grid(model, window, extractor, mean, std):
    """Predict the grid following ``window`` (seq_length, H, W), in degC.

    ``extractor`` is from src.data.extractors (extract_neighborhood for the CA framing, extract_pointwise for the no-CA one); both give cells in the same row-major interior order.
    the cell selection is the same as in training: the full 3x3 neighborhood must be finite over the whole window.

    Returns:
    * pred : (H, W) tensor, NaN wherever the model made no prediction.
    * predicted_mask : (H, W) bool tensor, True where the model predicted.
    """
    H, W = window.shape[1:] #dimensions of the grid since window will have seq_length at dim=0
    # target_map only enters the mask; window[-1] has the same land as any frame
    valid = neighborhood_valid_mask(window, window[-1]) #returns the cells that have 3*3 neigh non-nan
    X, _ = extractor(window, window[-1])
    X = (X[valid] - mean) / std

    interior = torch.full(((H - 2) * (W - 2),), float("nan"), device=window.device)  #it's gotta have the same shape as valid
    interior[valid] = model(X).squeeze(1) * std + mean    #now it can write the available values at the right place using the mask
    pred = torch.full((H, W), float("nan"), device=window.device) #full grid shape
    pred[1:-1, 1:-1] = interior.view(H - 2, W - 2) #writing values to their place in the grid, nan elsewhere

    predicted_mask = ~torch.isnan(pred)
    # a NaN model output would otherwise pass as "unpredicted" and be silently filled
    assert predicted_mask.sum() == valid.sum(), "model returned NaN on some cells"

    #pred contains actual values, nan where model could not predic
    #predicted_mask contains True wherever pred has a values, False elsewhere.
    return pred, predicted_mask



# STEP 3: Fill functions for the cells the models cannot predict
# All fill functions share one signature, fill(pred, predicted_mask, sea_mask, window, true_next),
# so the rollout can call whichever one it is given without knowing which it is.
# Each ignores the argument it doesn't need (ground_truth ignores window, neighbour_mean ignores true_next).

def fill_with_ground_truth(pred, predicted_mask, sea_mask, window, true_next):
    """Unpredicted sea cells (exterior ring, coast) take the real next values."""
    grid = pred.clone()
    to_fill = sea_mask & ~predicted_mask
    grid[to_fill] = true_next[to_fill]
    return grid


def fill_with_neighbour_mean(
    pred, predicted_mask, sea_mask, window, true_next=None, max_iter=4
):
    """Unpredicted sea cells take the mean of their available Moore neighbours.

    "Available" means non-NaN in the current grid, so land never contributes.
    Runs in passes: cells with no available neighbour (e.g. a corner whose
    neighbours are all unpredicted too) are filled once their neighbours are.
    Any cell still empty after ``max_iter`` passes keeps its previous value
    (window[-1]). Ground truth is never used.
    """
    grid = pred.clone()
    to_fill = sea_mask & ~predicted_mask
    H, W = grid.shape

    for _ in range(max_iter):
        if not to_fill.any():
            break
        padded = F.pad(grid[None, None], (1, 1, 1, 1), value=float("nan"))[0, 0]
        neigh = padded.unfold(0, 3, 1).unfold(1, 3, 1).reshape(H, W, 9)
        means = torch.nanmean(neigh, dim=-1)  # centre is NaN for cells to fill
        fillable = to_fill & ~torch.isnan(means)
        if not fillable.any():
            break
        grid[fillable] = means[fillable]
        to_fill &= ~fillable

    grid[to_fill] = window[-1][to_fill]
    return grid


FILL_FUNCTIONS = {
    "ground_truth": fill_with_ground_truth,
    "neighbour_mean": fill_with_neighbour_mean,
}


# Rollouts, one per input framing

def rollout_ca_rnn(
    model,
    sst_series,
    mean,
    std,
    fill_fn,
    start=0,
    n_steps=30,
    seq_length=SEQ_LENGTH,
):
    """RNN + CA rollout: seed with real frames [start, start+seq_length), then
    predict ``n_steps`` grids autoregressively.

    Returns
    -------
    preds : (n_steps, H, W) the filled grids that were fed back, in degC.
    truth : (n_steps, H, W) the real frames at the same times.
    predicted_masks : (n_steps, H, W) cells the model itself predicted.
    """
    first_target = start + seq_length
    if first_target + n_steps > sst_series.shape[0]:
        raise ValueError(
            f"Need {first_target + n_steps} frames, series has {sst_series.shape[0]}"
        )

    window = sst_series[start:first_target].clone()
    truth = sst_series[first_target : first_target + n_steps]
    sea_mask = ~torch.isnan(sst_series[start])   #all cells that had a value on the first grid are recognized as ocean cells

    preds, masks = [], []
    for k in range(n_steps):
        pred, predicted_mask = predict_next_grid(
            model, window, extract_neighborhood, mean, std
        )
        next_grid = fill_fn(pred, predicted_mask, sea_mask, window, truth[k])
        preds.append(next_grid)
        masks.append(predicted_mask)
        window = torch.cat([window[1:], next_grid.unsqueeze(0)]) #tu oublie pas de rajouter une dim

    return torch.stack(preds), truth, torch.stack(masks)


class NeighbourAverage(torch.nn.Module):
    """Model-free CA rule: the next value of a cell is the mean of its 3x3
    Moore neighbourhood (centre included) in the last frame of the window.
    Takes the ``extract_neighborhood`` input (n_cells, seq_length, 9)."""

    def forward(self, x):
        return x[:, -1, :].mean(dim=1, keepdim=True)


def rollout_neighbour_average(
    sst_series, fill_fn, start=0, n_steps=30, seq_length=SEQ_LENGTH
):
    """Baseline rollout where each step is the 3x3 neighbourhood average of the previous grid.
    Returns (preds, truth, predicted_masks), same as ``rollout_ca_rnn``.
    """
    return rollout_ca_rnn(
        NeighbourAverage(), sst_series, 0.0, 1.0, fill_fn,
        start=start, n_steps=n_steps, seq_length=seq_length,
    )


# Metrics for prediction error (accuracy)

def errors_per_step(preds, truth, cell_masks):
    """MAE and RMSE (degC) per step, over the masked cells only."""
    mae, rmse = [], []
    for p, t, m in zip(preds, truth, cell_masks):
        diff = p[m] - t[m]
        mae.append(diff.abs().mean().item())
        rmse.append(diff.pow(2).mean().sqrt().item())
    return {"mae": mae, "rmse": rmse}



# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--fill", default="neighbour_mean", choices=list(FILL_FUNCTIONS))
    p.add_argument("--start", type=int, default=0, help="Index of the first seed frame")
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--seq-length", type=int, default=SEQ_LENGTH)
    p.add_argument("--tag", default=None, help="Prefix for the output folder")
    return p.parse_args()


def main():
    args = parse_args()
    # CPU only: one small batch per step in a sequential loop, a GPU would not pay off
    model, mean, std, meta = load_checkpoint(args.checkpoint, "cpu")

    #following if to remove one "nn" and "ffnn" are implemented as well
    if meta["dataset"] != "nca":
        raise ValueError(
            f"rollout_ca_rnn needs a model trained on 'nca', got '{meta['dataset']}'"
        )

    sst = load_sst_series("test")
    preds, truth, masks = rollout_ca_rnn(
        model,
        sst,
        mean,
        std,
        FILL_FUNCTIONS[args.fill],
        start=args.start,
        n_steps=args.n_steps,
        seq_length=args.seq_length,
    )
    errors = errors_per_step(preds, truth, masks)

    tag = args.tag or f"ca_rnn_{meta['model_type'].lower()}_{args.fill}"
    run_dir = OUT_DIR / f"{tag}_{datetime.now().strftime('%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True)

    config = {
        "rollout": "ca_rnn",
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": meta,
        "split": "test",
        "fill": args.fill,
        "start": args.start,
        "n_steps": args.n_steps,
        "seq_length": args.seq_length,
        "grid_shape": list(sst.shape[1:]),
        "n_predicted_cells": int(masks[0].sum()),
        "n_sea_cells": int((~torch.isnan(sst[0])).sum()),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(run_dir / "errors.json", "w") as f:
        json.dump(errors, f, indent=2)
    torch.save(
        {"preds": preds.cpu(), "truth": truth.cpu(), "predicted_masks": masks.cpu()},
        run_dir / "rollout.pt",
    )

    print(f"MAE step 1: {errors['mae'][0]:.4f} | step {args.n_steps}: {errors['mae'][-1]:.4f}")
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
