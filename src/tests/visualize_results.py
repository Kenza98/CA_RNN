import torch
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
print(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"


def preds_vs_targets(Y, Y_model):
    # histogram of predictions vs targets
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(Y, bins=50, alpha=0.7, label="Targets")
    ax.hist(Y_model, bins=50, alpha=0.7, label="Predictions")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Distribution of Predictions vs Targets")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    save_file_path = OUT_DIR / "histogram_model_tar.png"
    fig.savefig(save_file_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_hist(Y_baseline, Y):
    # baseline histogram
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(Y_baseline, bins=50, alpha=0.7, label="Baseline")
    ax.hist(Y, bins=50, alpha=0.7, label="Targets")
    ax.legend()
    ax.set_title("Distribution of Predictions vs Targets")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    out_dir = os.path.dirname(__file__)
    save_file_path = OUT_DIR / "hist_model_baseline.png"
    fig.savefig(save_file_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_preds_vs_target(y_true, y_pred, fp):
    M = 300
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        y_true[:M], label="Target", marker="o", markersize=3, linestyle="-", alpha=0.7
    )
    ax.plot(
        y_pred[:M],
        label="Prediction",
        marker="x",
        markersize=3,
        linestyle="-",
        alpha=0.7,
    )
    ax.set_title("Zoomed-in: First 300 Test Samples")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.legend()
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)


def line_plot_mse(mse_vals, fp):
    # MSE plot comparaison
    labels = ["Baseline", "Model"]
    colors = ["mediumvioletred", "mediumaquamarine"]
    bar_width = 0.4
    x_pos = [0, 1]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x_pos, mse_vals, width=bar_width, color=colors, edgecolor="black")

    ax.set_ylabel("MSE", fontsize=12)

    # X-ticks and labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)

    # Annotate bars
    for bar, val in zip(bars, mse_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.1,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    # plt.tight_layout()
    ax.set_title("Mean Squared Error: Baseline vs Model", fontsize=14)
    fig.savefig(fp, dpi=150)
    plt.close(fig)


def line_plot_mare(mare_vals, fp):
    x_pos = [0, 1]
    labels = ["Baseline", "Model"]
    bar_width = 0.4
    colors = ["mediumvioletred", "mediumaquamarine"]

    # MARE plot comparaison
    fig, ax = plt.subplots()
    bars = ax.bar(x_pos, mare_vals, width=bar_width, color=colors, edgecolor="black")

    # X-ticks and labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)

    ax.set_ylabel("MARE", fontsize=12)
    # Annotate bars
    for bar, val in zip(bars, mare_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.1,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.set_title("Mean Absolute Relative Error: Baseline vs Model")
    fig.savefig(fp, dpi=150)
    plt.close(fig)

def plot_mse_boxplot(model_sample_error, baseline_sample_error, fp):
    fig,axes = plt.subplots()
    axes.boxplot([
        baseline_sample_error,
        model_sample_error,
    ],
    tick_labels=["Baseline", "Model"],
    showfliers=False,
    )
    axes.set_title("MSE Error Distribution")
    axes.set_ylabel("Square Error")
    axes.grid(True, axis="y", linestyle="--", alpha=.5)
    fig.suptitle("Error Distribution: Baseline vs Model")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_mare_boxplot(model_sample_error, baseline_sample_error, fp):
    fig,axes = plt.subplots()
    axes.boxplot([
        baseline_sample_error,
        model_sample_error,
    ],
    tick_labels=["Baseline", "Model"],
    showfliers=False,
    )
    axes.set_title("MARE Error Distribution")
    axes.set_ylabel("Mean Absolute Relative Error")
    axes.grid(True, axis="y", linestyle="--", alpha=.5)
    fig.suptitle("Error Distribution: Baseline vs Model")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)



def main():
    output_file = OUT_DIR / "testset_results.pt"
    results = torch.load(output_file, map_location="cpu")

    mse_model_all = results["mse_model_all"]
    mse_baseline_all = results["mse_baseline_all"]
    mare_model_all = results["mare_model_all"]
    mare_baseline_all = results["mare_baseline_all"]

    # recompute averages
    mse = mse_model_all.mean().item()
    baseline_mse = mse_baseline_all.mean().item()

    mare = mare_model_all.mean().item()
    baseline_mare = mare_baseline_all.mean().item()

    mse_vals = [baseline_mse, mse]
    mare_vals = [baseline_mare, mare]

    # print(f"Average MSE: {mse:.4f}, Average MARE: {mare:.4f}")
    # print(f"Baseline MSE: {baseline_mse:.4f}, Baseline MARE: {baseline_mare:.4f}")

    # fp_mse = OUT_DIR / "vrnn_bar_mse_comparison.png"
    # line_plot_mse(mse_vals, fp_mse)

    # fp_mare = OUT_DIR / "vrnn_bar_mare_comparison.png"
    # line_plot_mare(mare_vals, fp_mare)

    #fp_box = OUT_DIR / "vrnn_mse_boxplot.png"
    # plot_error_boxplot(
    #     mse_model_all, mse_baseline_all, mare_model_all, mare_baseline_all, fp_box
    # )
    #plot_mse_boxplot(mse_baseline_all, mse_model_all, fp_box)

    mare_boxplot = OUT_DIR / "vrnn_mare_boxplot.png"
    plot_mare_boxplot(mare_model_all, mare_baseline_all, mare_boxplot)




if __name__ == "__main__":
    main()
