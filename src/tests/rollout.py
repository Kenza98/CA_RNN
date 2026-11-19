"""Autoregressive rollout of trained models over the full SST grid.

A rollout starts from ``seq_length`` real frames, predicts the next grid,
appends it to the window and repeats for ``n_steps``.
After the seed, the model never sees a real frame, unless a fill function puts one in.

Contents:
- Loading: ``load_sst_series``, ``load_checkpoint``.
- One step: ``predict_next_grid``. Cells the model cannot predict
  (land, outer ring, coast touching land) are left as NaN.
- Fill functions, for those unpredicted sea cells, before the grid is fed back:
  ``fill_with_ground_truth`` (real values) and ``fill_with_neighbour_mean`` (no real values).
- Rollouts: ``rollout_ca_rnn`` (RNN + CA), ``rollout_rnn`` (RNN without CA),
  ``rollout_neighbour_average`` and ``rollout_persistence`` (baselines, no model).
  \\TODO ``rollout_ffnn``.
- Errors: ``errors_per_step``, on the cells the model predicted only.

To run the rollouts and plot their errors, see ``compute_rollout_errors.py`` and ``plot_rollout.py``.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import xarray as xr

from src.data.download import raw_path
from src.data.extractors import (
    extract_neighborhood,
    extract_pointwise,
    neighborhood_valid_mask,
)
from src.models.gru import GRU
from src.models.lstm import LSTM
from src.models.VanillaRNN import VanillaRNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "ca_results"

SEQ_LENGTH = 6
MODEL_CLASSES = {"GRU": GRU, "LSTM": LSTM, "VanillaRNN": VanillaRNN}

# number of input features per cell, matching the extractors in src/data/extractors.py
INPUT_DIMS = {"nca": 9, "nn": 1}


# Loading


def load_sst_series(split="test"):
    """Raw SST time series of a split, as a float tensor (T, H, W) in degC."""
    with xr.open_dataset(raw_path(split)) as ds:
        sst = ds["thetao"]
        if "depth" in sst.dims:
            sst = sst.isel(depth=0)
        return torch.from_numpy(sst.values).float()


def load_checkpoint(path, device):
    """Rebuild a trained model from its checkpoint.

    Returns (model, mean, std, meta): mean and std are the train-set statistics
    the model was trained with, meta holds the checkpoint's hyperparameters.
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

    meta = {key: value for key, value in ckpt.items() if not key.endswith("StateDict")}
    return model, float(ckpt["global_mean"]), float(ckpt["global_std"]), meta


# One step: window -> next grid


@torch.no_grad()
def predict_next_grid(model, window, extractor, mean, std):
    """Predict the grid that follows ``window`` (seq_length, H, W), in degC.

    ``extractor`` is ``extract_neighborhood`` (with CA) or ``extract_pointwise`` (without CA).
    Cells are selected as in training: the full 3x3 neighbourhood must be finite over the whole window.

    Returns:
    * pred: (H, W) tensor, NaN wherever the model made no prediction.
    * predicted_mask: (H, W) bool tensor, True where the model predicted.
    """
    # window is (seq_length, H, W)
    H, W = window.shape[1:]

    # cells whose 3x3 neighbourhood has no NaN, in the interior's row-major order.
    # window[-1] is passed as target: it has the same land as any frame
    valid = neighborhood_valid_mask(window, window[-1])

    X, _ = extractor(window, window[-1])
    X = (X[valid] - mean) / std
    predictions = model(X).squeeze(1) * std + mean

    # write the predictions back at their place in the interior, NaN elsewhere
    interior = torch.full(((H - 2) * (W - 2),), float("nan"), device=window.device)
    interior[valid] = predictions

    # then the interior into the full grid; the outer ring stays NaN
    pred = torch.full((H, W), float("nan"), device=window.device)
    pred[1:-1, 1:-1] = interior.view(H - 2, W - 2)

    predicted_mask = ~torch.isnan(pred)

    # a NaN model output would otherwise pass as "unpredicted" and be silently filled
    assert predicted_mask.sum() == valid.sum(), "model returned NaN on some cells"

    return pred, predicted_mask


# Fill functions for the cells the model cannot predict
# All share one signature, fill(pred, predicted_mask, sea_mask, window, true_next),
# so a rollout can call whichever one it is given.
# Each ignores the argument it does not need.


def fill_with_ground_truth(pred, predicted_mask, sea_mask, window, true_next):
    """Unpredicted sea cells (outer ring, coast) take the real next values."""
    grid = pred.clone()
    to_fill = sea_mask & ~predicted_mask
    grid[to_fill] = true_next[to_fill]
    return grid


def fill_with_neighbour_mean(pred, predicted_mask, sea_mask, window, true_next=None, max_iter=4):
    """Unpredicted sea cells take the mean of their non-NaN 3x3 neighbours (land never counts).

    Runs in passes, so a cell with no available neighbour is filled once its neighbours are.
    A cell still empty after ``max_iter`` passes keeps its previous value, window[-1].
    Real values are never used.
    """
    grid = pred.clone()
    to_fill = sea_mask & ~predicted_mask
    H, W = grid.shape

    for _ in range(max_iter):
        if not to_fill.any():
            break

        # 3x3 neighbourhood of every cell, with NaN padding around the grid
        padded = F.pad(grid[None, None], (1, 1, 1, 1), value=float("nan"))[0, 0]
        neighbours = padded.unfold(0, 3, 1).unfold(1, 3, 1).reshape(H, W, 9)

        # the centre is NaN for the cells to fill, so it never counts
        means = torch.nanmean(neighbours, dim=-1)

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


def rollout_ca_rnn(model, sst_series, mean, std, fill_fn, start=0, n_steps=30, seq_length=SEQ_LENGTH):
    """RNN + CA rollout: seed with the real frames [start, start + seq_length),
    then predict ``n_steps`` grids, each fed back into the window.

    Returns:
    * preds: (n_steps, H, W) the filled grids that were fed back, in degC.
    * truth: (n_steps, H, W) the real frames at the same times.
    * predicted_masks: (n_steps, H, W) the cells the model itself predicted.
    """
    # check that the series has enough frames for the seed and every step
    first_target = start + seq_length
    if first_target + n_steps > sst_series.shape[0]:
        raise ValueError(f"Need {first_target + n_steps} frames, series has {sst_series.shape[0]}")

    window = sst_series[start:first_target].clone()
    truth = sst_series[first_target : first_target + n_steps]

    # cells with a value on the first frame are sea
    sea_mask = ~torch.isnan(sst_series[start])

    preds, masks = [], []
    for k in range(n_steps):
        pred, predicted_mask = predict_next_grid(model, window, extract_neighborhood, mean, std)
        next_grid = fill_fn(pred, predicted_mask, sea_mask, window, truth[k])

        preds.append(next_grid)
        masks.append(predicted_mask)

        # drop the oldest frame, append the new one (unsqueeze: tu oublies pas de rajouter une dim)
        window = torch.cat([window[1:], next_grid.unsqueeze(0)])

    return torch.stack(preds), truth, torch.stack(masks)


def rollout_rnn(model, sst_series, mean, std, fill_fn, start=0, n_steps=30, seq_length=SEQ_LENGTH):
    """RNN rollout without CA: each cell is predicted from its own history only.

    Same seed, cells, fill and feedback as ``rollout_ca_rnn``, so the two are compared on the same cells.
    Returns (preds, truth, predicted_masks), as ``rollout_ca_rnn``.
    """
    # check that the series has enough frames for the seed and every step
    first_target = start + seq_length
    if first_target + n_steps > sst_series.shape[0]:
        raise ValueError(f"Need {first_target + n_steps} frames, series has {sst_series.shape[0]}")

    # seed window, and the real frames to compare with (also used by fill_with_ground_truth)
    window = sst_series[start:first_target].clone()
    truth = sst_series[first_target : first_target + n_steps]

    # cells with a value on the first frame are sea
    sea_mask = ~torch.isnan(sst_series[start])

    # lists first, stacked into tensors at the end
    preds, masks = [], []
    for k in range(n_steps):
        pred, predicted_mask = predict_next_grid(model, window, extract_pointwise, mean, std)
        next_grid = fill_fn(pred, predicted_mask, sea_mask, window, truth[k])

        preds.append(next_grid)
        masks.append(predicted_mask)

        # drop the oldest frame, append the new one (unsqueeze: tu oublies pas de rajouter une dim)
        window = torch.cat([window[1:], next_grid.unsqueeze(0)])

    # predictions, real frames (for the errors), masks (for the errors and the maps)
    return torch.stack(preds), truth, torch.stack(masks)


class NeighbourAverage(torch.nn.Module):
    """Model-free CA rule: a cell's next value is the mean of its 3x3 neighbourhood
    (centre included) in the last frame. Takes ``extract_neighborhood`` input (n_cells, seq_length, 9)."""

    def forward(self, x):
        last_frame = x[:, -1, :]
        return last_frame.mean(dim=1, keepdim=True)


def rollout_neighbour_average(sst_series, fill_fn, start=0, n_steps=30, seq_length=SEQ_LENGTH):
    """Baseline rollout: each step is the 3x3 neighbourhood average of the previous grid.

    Runs through ``rollout_ca_rnn``, so it uses the same seed, cells and fill as the model.
    mean=0 and std=1 because there is no trained model to standardise for.
    Returns (preds, truth, predicted_masks), as ``rollout_ca_rnn``.
    """
    return rollout_ca_rnn(
        NeighbourAverage(),
        sst_series,
        mean=0.0,
        std=1.0,
        fill_fn=fill_fn,
        start=start,
        n_steps=n_steps,
        seq_length=seq_length,
    )


class Persistence(torch.nn.Module):
    """Model-free rule: a cell's next value is its own value in the last frame.
    Takes ``extract_neighborhood`` input (n_cells, seq_length, 9); index 4 is the centre cell."""

    def forward(self, x):
        return x[:, -1, 4:5]


def rollout_persistence(sst_series, fill_fn, start=0, n_steps=30, seq_length=SEQ_LENGTH):
    """Baseline rollout: each step repeats the previous grid (on the predicted cells).

    Runs through ``rollout_ca_rnn``, so it uses the same seed, cells and fill as the model.
    Returns (preds, truth, predicted_masks), as ``rollout_ca_rnn``.
    """
    return rollout_ca_rnn(
        Persistence(),
        sst_series,
        mean=0.0,
        std=1.0,
        fill_fn=fill_fn,
        start=start,
        n_steps=n_steps,
        seq_length=seq_length,
    )


# Errors


def errors_per_step(preds, truth, cell_masks):
    """MAE and RMSE (degC) per step, over the masked cells only."""
    mae, rmse = [], []
    for pred, true, mask in zip(preds, truth, cell_masks):
        diff = pred[mask] - true[mask]
        mae.append(diff.abs().mean().item())
        rmse.append(diff.pow(2).mean().sqrt().item())
    return {"mae": mae, "rmse": rmse}
