"""Run the rollouts of a checkpoint on the test set and save their errors, per start and per step.

Runs the model's rollout (``nca``: with CA, ``nn``: without CA) and the two baselines
(neighbour average and persistence) from every start (or one, with ``--start``), and saves MAE and RMSE to a JSON file.
``plot_rollout.py`` plots that file.

Usage:
    python -m src.tests.compute_rollout_errors --checkpoint models/lstm_1_nca_cpu_0917_1447.pt
    python -m src.tests.compute_rollout_errors --checkpoint ... --start 30 --fill ground_truth
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.tests.rollout import (
    FILL_FUNCTIONS,
    OUT_DIR,
    SEQ_LENGTH,
    errors_per_step,
    load_checkpoint,
    load_sst_series,
    rollout_ca_rnn,
    rollout_neighbour_average,
    rollout_persistence,
    rollout_rnn,
)

# which rollout to use, from the checkpoint's dataset
ROLLOUTS = {"nca": rollout_ca_rnn, "nn": rollout_rnn}
FRAMING_LABELS = {"nca": "with CA", "nn": "without CA"}

ERRORS_DIR = OUT_DIR / "rollout_errors"


def compute_rollout_errors(checkpoint, fill="neighbour_mean", start=None, n_steps=30, seq_length=SEQ_LENGTH):
    """Errors of the model and of the two baselines, from one start or from all of them.

    Returns a JSON-ready dict. ``errors[name][metric]`` is a list over starts of lists over steps,
    with name "model", "neighbour_average" or "persistence" and metric "mae" or "rmse".
    """
    model, mean, std, meta = load_checkpoint(checkpoint, "cpu")
    dataset = meta["dataset"]
    if dataset not in ROLLOUTS:
        raise ValueError(f"No rollout for dataset '{dataset}', expected one of {list(ROLLOUTS)}")

    model_rollout = ROLLOUTS[dataset]
    fill_fn = FILL_FUNCTIONS[fill]

    sst = load_sst_series("test")
    if start is not None:
        starts = [start]
    else:
        n_starts = sst.shape[0] - seq_length - n_steps + 1
        starts = list(range(n_starts))

    errors = {
        "model": {"mae": [], "rmse": []},
        "neighbour_average": {"mae": [], "rmse": []},
        "persistence": {"mae": [], "rmse": []},
    }

    for s in starts:
        model_run = model_rollout(
            model, sst, mean, std, fill_fn, start=s, n_steps=n_steps, seq_length=seq_length
        )
        average_run = rollout_neighbour_average(
            sst, fill_fn, start=s, n_steps=n_steps, seq_length=seq_length
        )
        persistence_run = rollout_persistence(
            sst, fill_fn, start=s, n_steps=n_steps, seq_length=seq_length
        )

        runs = {
            "model": model_run,
            "neighbour_average": average_run,
            "persistence": persistence_run,
        }
        for name, (preds, truth, masks) in runs.items():
            step_errors = errors_per_step(preds, truth, masks)
            errors[name]["mae"].append(step_errors["mae"])
            errors[name]["rmse"].append(step_errors["rmse"])

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_meta": meta,
        "model_label": f"{meta['model_type']} {FRAMING_LABELS[dataset]}",
        "split": "test",
        "fill": fill,
        "n_steps": n_steps,
        "seq_length": seq_length,
        "starts": starts,
        "errors": errors,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--fill", default="neighbour_mean", choices=list(FILL_FUNCTIONS))
    p.add_argument("--start", type=int, default=None, help="Single start frame (default: every possible start)")
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--seq-length", type=int, default=SEQ_LENGTH)
    p.add_argument("--out", type=Path, default=None, help="Output .json (default: timestamped, in outputs/ca_results/rollout_errors/)")
    return p.parse_args()


def main():
    args = parse_args()
    result = compute_rollout_errors(args.checkpoint, args.fill, args.start, args.n_steps, args.seq_length)

    if args.out is not None:
        out_path = args.out
    else:
        which = f"start{args.start}" if args.start is not None else "allstarts"
        stamp = datetime.now().strftime("%m%d_%H%M%S")
        out_path = ERRORS_DIR / f"{args.checkpoint.stem}_{args.fill}_{which}_{stamp}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f)

    print(f"{len(result['starts'])} start(s), saved to {out_path}")


if __name__ == "__main__":
    main()
