"""Plot the rollout errors saved by ``compute_rollout_errors.py``, as MAE, RMSE or MSE, for the thesis.

Two plots from the same file:
- ``plot_all_starts``: across all starts, at each step: mean, median, interquartile range and min-max range.
- ``plot_single_start``: the curve of one start (``--start N``, or ``--single-start`` for a random one).
``--baseline average|persistence|all`` adds baselines in grey, one line style each
(several can be given, ``all`` is both); ``--col`` sets the model colour.

Figures have no title (the caption is written in LaTeX) and use Computer Modern, LaTeX's default font.
Each figure is saved as .pdf (vector, for LaTeX) and .png.

Usage:
    python -m src.tests.plot_rollout outputs/ca_results/rollout_errors/<file>.json --metric rmse --baseline all
    python -m src.tests.plot_rollout outputs/ca_results/rollout_errors/<file>.json --metric mae --start 30
    python -m src.tests.plot_rollout outputs/ca_results/rollout_errors/<file>.json --metric mae --single-start --baseline persistence
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, e.g. on the cluster
import matplotlib.pyplot as plt
import torch

# axis labels, in mathtext so the degree sign is drawn in Computer Modern too
Y_LABELS = {
    "mae": r"MAE ($^{\circ}$C)",
    "rmse": r"RMSE ($^{\circ}$C)",
    "mse": r"MSE ($^{\circ}$C$^{2}$)",
}

# model colour, chosen with --col (pink by default); baseline in grey
MODEL_COLORS = {
    "pink": "#d55181",
    "yellow": "#cf9f1f",     # mustard
    "turquoise": "#1b9db3",  # turquoise blue
    "duck": "#0b6b66",       # duck green
}
BASELINE_COLOR = "#8c8b85"

# --baseline choice -> (key in the errors file, legend label, line style); all in BASELINE_COLOR
BASELINES = {
    "average": ("neighbour_average", "Neighbour average", "-"),
    "persistence": ("persistence", "Persistence", ":"),
}
BASELINE_CHOICES = [*BASELINES, "all"]


def resolve_baselines(choices):
    """--baseline values -> baseline names, in BASELINES order, without repeats;"all" is every baseline."""
    if not choices:
        return [] # No baselines arg, doesn't plot baselines
    if "all" in choices:
        return list(BASELINES) #if "all" is listed or is the only arg, show all
    # Finally reorder requested baselines and remove repeats
    return [name for name in BASELINES if name in choices]


def baseline_suffix(baselines):
    """Filename suffix for the plotted baselines: "", "_average", "_persistence" or "_all"."""
    if not baselines:
        return ""
    if set(baselines) == set(BASELINES):
        return "_all"
    return "_" + "_".join(baselines)


def plot_baselines(ax, result, metric, baselines, reduce, **line_kwargs):
    """One grey line per baseline, ``reduce`` turning its (n_starts, n_steps) values into a curve.
    Returns the (handle, label) legend pairs."""
    steps = torch.arange(1, result["n_steps"] + 1)
    legend_items = []
    for name in baselines:
        key, label, linestyle = BASELINES[name]
        values = reduce(metric_values(result, key, metric))
        (line,) = ax.plot(steps, values, color=BASELINE_COLOR, linewidth=1.4, linestyle=linestyle, **line_kwargs)
        legend_items.append((line, label))
    return legend_items

# axes and grid colours
INK = "#1f1f1e"
GRID = "#e6e5e0"

# about the text width of an A4 thesis page, so 10 pt text stays 10 pt once included
FIGSIZE = (5.9, 3.4)

THESIS_STYLE = {
    # Computer Modern, bundled with matplotlib, so no LaTeX install is needed
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # cmr10 has no minus sign, mathtext does
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed fonts as TrueType, so the PDF text stays selectable
}


def metric_values(result, name, metric):
    """(n_starts, n_steps) tensor of ``metric`` for ``name`` ("model", "neighbour_average" or "persistence").
    MSE is each rollout's RMSE squared."""
    if name not in result["errors"]:
        raise KeyError(f"No '{name}' errors in this file: recompute it with compute_rollout_errors.py")
    if metric == "mse":
        rmse = torch.tensor(result["errors"][name]["rmse"])
        return rmse**2
    return torch.tensor(result["errors"][name][metric])


def new_figure():
    plt.rcParams.update(THESIS_STYLE)
    return plt.subplots(figsize=FIGSIZE)


def style_and_save(fig, ax, metric, n_steps, legend_items, out_path):
    """Axes shared by both plots, legend from (handle, label) pairs in the given order,
    then save as .pdf and .png. Returns the two paths."""
    ax.set_xlabel("Rollout step (days)")
    ax.set_ylabel(Y_LABELS[metric])

    ax.set_xlim(1, n_steps)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [handle for handle, _ in legend_items]
    labels = [label for _, label in legend_items]
    ax.legend(handles, labels, frameon=False, loc="upper left")

    pdf_path = Path(out_path).with_suffix(".pdf")
    png_path = Path(out_path).with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path, png_path


def plot_all_starts(result, metric, out_path, baselines=(), color=MODEL_COLORS["pink"]):
    """Across all starts, at each step: mean, median, interquartile range and min-max range.
    ``baselines`` (names from BASELINES) adds each baseline's mean in grey."""
    steps = torch.arange(1, result["n_steps"] + 1)
    values = metric_values(result, "model", metric)

    fig, ax = new_figure()

    # bands first, lightest at the back, then the lines on top
    span = ax.fill_between(
        steps,
        values.min(dim=0).values,
        values.max(dim=0).values,
        color=color,
        alpha=0.10,
        linewidth=0,
    )
    quartiles = ax.fill_between(
        steps,
        values.quantile(0.25, dim=0),
        values.quantile(0.75, dim=0),
        color=color,
        alpha=0.28,
        linewidth=0,
    )

    baseline_items = plot_baselines(ax, result, metric, baselines, lambda v: v.mean(dim=0))

    (median_line,) = ax.plot(steps, values.median(dim=0).values, color=color, linewidth=1.4, linestyle="--")
    (mean_line,) = ax.plot(steps, values.mean(dim=0), color=color, linewidth=1.8)

    legend_items = [
        (mean_line, "Mean"),
        (median_line, "Median"),
        (quartiles, "Interquartile range"),
        (span, "Min-max range"),
    ]
    legend_items += [(line, f"{label} (mean)") for line, label in baseline_items]

    return style_and_save(fig, ax, metric, result["n_steps"], legend_items, out_path)


def plot_single_start(result, metric, start, out_path, baselines=(), color=MODEL_COLORS["pink"]):
    """The rollout curve of one start."""
    if start not in result["starts"]:
        first, last = result["starts"][0], result["starts"][-1]
        raise ValueError(f"Start {start} is not in this file (starts {first} to {last})")

    row = result["starts"].index(start)
    steps = torch.arange(1, result["n_steps"] + 1)

    fig, ax = new_figure()

    baseline_items = plot_baselines(ax, result, metric, baselines, lambda v: v[row])

    model_values = metric_values(result, "model", metric)[row]
    (model_line,) = ax.plot(steps, model_values, color=color, linewidth=1.8)

    legend_items = [(model_line, result["model_label"])] + baseline_items

    return style_and_save(fig, ax, metric, result["n_steps"], legend_items, out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("errors_file", type=Path, help="JSON written by compute_rollout_errors.py")
    p.add_argument("--metric", default="mae", choices=list(Y_LABELS))
    # --start N plots that start, --single-start plots a random one; not both
    which_start = p.add_mutually_exclusive_group()
    which_start.add_argument("--start", type=int, default=None, help="Plot this single start instead of all starts")
    which_start.add_argument("--single-start", action="store_true", help="Plot one start picked at random from the file")
    p.add_argument("--col", default="pink", choices=list(MODEL_COLORS), help="Colour of the model's curves and bands")
    p.add_argument(
        "--baseline", nargs="+", default=[], choices=BASELINE_CHOICES,
        help="Also plot these baselines in grey, one line style each; 'all' is every baseline",
    )
    p.add_argument("--out", type=Path, default=None, help="Output path, saved as .pdf and .png (default: next to the errors file)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.errors_file) as f:
        result = json.load(f)

    start = args.start
    baselines = resolve_baselines(args.baseline)

    # random start among the ones in the file; printed so the plot can be redone with --start
    if args.single_start:
        start = random.choice(result["starts"])
        print(f"Random start: {start}")

    # a file with a single start has no spread to plot: plot that start
    if start is None and len(result["starts"]) == 1:
        start = result["starts"][0]

    if args.out is not None:
        out_path = args.out
    else:
        which = f"start{start}" if start is not None else "allstarts"
        name = f"{args.errors_file.stem}_{args.metric}_{which}{baseline_suffix(baselines)}"
        out_path = args.errors_file.with_name(name)

    if start is not None:
        pdf_path, png_path = plot_single_start(result, args.metric, start, out_path, baselines, MODEL_COLORS[args.col])
    else:
        pdf_path, png_path = plot_all_starts(result, args.metric, out_path, baselines, MODEL_COLORS[args.col])

    print(f"Saved to {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
