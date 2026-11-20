"""Compare checkpoints: the all-starts mean error of each one on the same plot.

Takes several files written by ``compute_rollout_errors.py`` and makes one plot per framing
(``nca``: with CA, ``nn``: without CA). Each model type has its own colour (``MODEL_TYPE_COLORS``),
and ``--baseline average|persistence`` adds that baseline's mean in grey.
Same style as ``plot_rollout.py``: no title, Computer Modern, saved as .pdf and .png.

Usage:
    python -m src.tests.plot_rollout_comparison outputs/ca_results/rollout_errors/*_ground_truth_allstarts.json \\
        --metric mae --baseline persistence
"""

import argparse
import json
from pathlib import Path

import torch

from src.tests.plot_rollout import (
    BASELINE_COLOR,
    BASELINES,
    MODEL_COLORS,
    Y_LABELS,
    metric_values,
    new_figure,
    style_and_save,
)

# one colour per model type; the order is also the legend order
MODEL_TYPE_COLORS = {
    "GRU": MODEL_COLORS["pink"],
    "LSTM": MODEL_COLORS["duck"],
    "VanillaRNN": MODEL_COLORS["yellow"],
}

FRAMINGS = ["nca", "nn"]


def check_comparable(results):
    """All files must come from the same rollout setup, or their curves cannot share a plot."""
    first = results[0]
    for result in results[1:]:
        for key in ["fill", "n_steps", "seq_length", "starts"]:
            if result[key] != first[key]:
                raise ValueError(
                    f"'{key}' differs between {first['checkpoint']} and {result['checkpoint']}"
                )


def plot_comparison(results, metric, out_path, baseline=None):
    """Mean over starts of each result, one line per model type, plus the baseline's mean in grey.

    The baseline does not depend on the model, so it is taken from the first result.
    """
    model_types = [result["checkpoint_meta"]["model_type"] for result in results]
    if len(set(model_types)) < len(model_types):
        raise ValueError(f"Several checkpoints of the same model type: {model_types}")

    # fixed legend order: GRU, LSTM, VanillaRNN
    results = sorted(results, key=lambda r: list(MODEL_TYPE_COLORS).index(r["checkpoint_meta"]["model_type"]))

    n_steps = results[0]["n_steps"]
    steps = torch.arange(1, n_steps + 1)

    fig, ax = new_figure()
    legend_items = []

    for result in results:
        model_type = result["checkpoint_meta"]["model_type"]
        mean = metric_values(result, "model", metric).mean(dim=0)
        (line,) = ax.plot(steps, mean, color=MODEL_TYPE_COLORS[model_type], linewidth=1.8)
        legend_items.append((line, model_type))

    if baseline is not None:
        baseline_key, baseline_label = BASELINES[baseline]
        baseline_mean = metric_values(results[0], baseline_key, metric).mean(dim=0)
        # zorder 1: behind the model lines (which are at 2)
        (baseline_line,) = ax.plot(steps, baseline_mean, color=BASELINE_COLOR, linewidth=1.4, zorder=1)
        legend_items.append((baseline_line, baseline_label))

    return style_and_save(fig, ax, metric, n_steps, legend_items, out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("errors_files", type=Path, nargs="+", help="JSON files written by compute_rollout_errors.py")
    p.add_argument("--metric", default="mae", choices=list(Y_LABELS))
    p.add_argument("--baseline", default=None, choices=list(BASELINES), help="Also plot this baseline, in grey")
    p.add_argument("--out-dir", type=Path, default=None, help="Output folder (default: the folder of the first file)")
    return p.parse_args()


def main():
    args = parse_args()

    results = []
    for path in args.errors_files:
        with open(path) as f:
            results.append(json.load(f))

    check_comparable(results)

    out_dir = args.out_dir if args.out_dir is not None else args.errors_files[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # one plot per framing, with the checkpoints of that framing
    for framing in FRAMINGS:
        group = [r for r in results if r["checkpoint_meta"]["dataset"] == framing]
        if not group:
            continue

        name = f"comparison_{framing}_{group[0]['fill']}_{args.metric}"
        if args.baseline is not None:
            name += f"_{args.baseline}"

        pdf_path, png_path = plot_comparison(group, args.metric, out_dir / name, args.baseline)
        print(f"{framing}: {len(group)} checkpoint(s), saved to {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
