"""Plot the rollout errors saved by ``compute_rollout_errors.py``, as MAE, RMSE or MSE.

Two plots from the same file:
- ``plot_all_starts``: mean over all starts at each step, with the 25-75 percentile band.
- ``plot_single_start``: the curve of one start (``--start``).
``--baseline`` adds the neighbour-average rollout in grey, behind the model.

Usage:
    python -m src.tests.plot_rollout outputs/ca_results/rollout_errors/<file>.json --metric rmse --baseline
    python -m src.tests.plot_rollout outputs/ca_results/rollout_errors/<file>.json --metric mae --start 30
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, e.g. on the cluster
import matplotlib.pyplot as plt
import torch

UNITS = {"mae": "°C", "rmse": "°C", "mse": "°C²"}

# model in pink, baseline in grey (drawn behind)
MODEL_COLOR = "#d55181"
BASELINE_COLOR = "#a3a29b"

# text, axes and grid colours
INK = "#1f1f1e"
MUTED = "#6b6a64"
GRID = "#e4e3de"


def metric_values(result, name, metric):
    """(n_starts, n_steps) tensor of ``metric`` for ``name`` ("model" or "neighbour_average").
    MSE is each rollout's RMSE squared."""
    if metric == "mse":
        rmse = torch.tensor(result["errors"][name]["rmse"])
        return rmse**2
    return torch.tensor(result["errors"][name][metric])


def curves_to_plot(result, metric, baseline):
    """List of (label, color, values). The baseline comes first, so it is drawn behind the model."""
    curves = []
    if baseline:
        baseline_values = metric_values(result, "neighbour_average", metric)
        curves.append(("Neighbour average", BASELINE_COLOR, baseline_values))

    model_values = metric_values(result, "model", metric)
    curves.append((result["model_label"], MODEL_COLOR, model_values))
    return curves


def draw_line(ax, steps, values, label, color, zorder):
    """The curve, with its last value written at the end of the line."""
    ax.plot(steps, values, color=color, linewidth=2, label=label, zorder=zorder)
    ax.annotate(
        f"{values[-1]:.2f}",
        (steps[-1], values[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=INK,
    )


def style_and_save(fig, ax, metric, title, n_steps, out_path):
    """Axes, labels and legend shared by both plots, then save the figure."""
    ax.set_xlabel("Rollout step (days)", color=INK)
    ax.set_ylabel(f"{metric.upper()} ({UNITS[metric]})", color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")

    ax.set_xlim(1, n_steps)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)

    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_all_starts(result, metric, out_path, baseline=False):
    """Mean over starts at each step, with the 25-75 percentile band across starts."""
    steps = torch.arange(1, result["n_steps"] + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)

    for z, (label, color, values) in enumerate(curves_to_plot(result, metric, baseline)):
        q25 = values.quantile(0.25, dim=0)
        q75 = values.quantile(0.75, dim=0)

        # bands at zorder < 10 and lines at zorder >= 10: no band covers a line
        ax.fill_between(steps, q25, q75, color=color, alpha=0.18, linewidth=0, zorder=z)
        draw_line(ax, steps, values.mean(dim=0), label, color, zorder=10 + z)

    n_starts = len(result["starts"])
    title = f"Rollout {metric.upper()} on the test set (mean over {n_starts} starts, band = 25-75th pct)"
    style_and_save(fig, ax, metric, title, result["n_steps"], out_path)


def plot_single_start(result, metric, start, out_path, baseline=False):
    """The rollout curve of one start."""
    if start not in result["starts"]:
        first, last = result["starts"][0], result["starts"][-1]
        raise ValueError(f"Start {start} is not in this file (starts {first} to {last})")

    row = result["starts"].index(start)
    steps = torch.arange(1, result["n_steps"] + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)

    for z, (label, color, values) in enumerate(curves_to_plot(result, metric, baseline)):
        draw_line(ax, steps, values[row], label, color, zorder=10 + z)

    title = f"Rollout {metric.upper()} on the test set (start {start})"
    style_and_save(fig, ax, metric, title, result["n_steps"], out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("errors_file", type=Path, help="JSON written by compute_rollout_errors.py")
    p.add_argument("--metric", default="mae", choices=list(UNITS))
    p.add_argument("--start", type=int, default=None, help="Plot this single start instead of the mean over starts")
    p.add_argument("--baseline", action="store_true", help="Also plot the neighbour-average rollout, in grey")
    p.add_argument("--out", type=Path, default=None, help="Output .png (default: next to the errors file)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.errors_file) as f:
        result = json.load(f)

    # a file with a single start has no mean to plot: plot that start
    start = args.start
    if start is None and len(result["starts"]) == 1:
        start = result["starts"][0]

    if args.out is not None:
        out_path = args.out
    else:
        which = f"start{start}" if start is not None else "mean"
        out_path = args.errors_file.with_name(f"{args.errors_file.stem}_{args.metric}_{which}.png")

    if start is not None:
        plot_single_start(result, args.metric, start, out_path, args.baseline)
    else:
        plot_all_starts(result, args.metric, out_path, args.baseline)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
