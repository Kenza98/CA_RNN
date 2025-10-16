import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

# Create a unique log directory for this run
log_dir = os.path.join("runs", f"rnnca_job_{os.environ.get('SLURM_JOB_ID', 'local')}")
writer = SummaryWriter(log_dir=log_dir)

output_file = "testset_results.pt"
results = torch.load(output_file, map_location="cpu")

Y = results["Y_true"]
Y_model = results["Y_model"]
Y_baseline = results["Y_baseline"]

"""
# histogram of predictions vs targets
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(Y, bins=50, alpha=0.7, label="Targets")
ax.hist(Y_model, bins=50, alpha=0.7, label="Predictions")
ax.set_yscale('log')
ax.legend()
ax.set_title("Distribution of Predictions vs Targets")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
out_dir = os.path.dirname(__file__)
fig.savefig(
    os.path.join(out_dir, "histogram_model_tar.png"), dpi=150, bbox_inches="tight"
)
writer.add_figure("Histogram of Model prediction vs Target", fig)
plt.close(fig)
"""
#baseline histogram
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(Y_baseline, bins=50, alpha=0.7, label="Baseline")
ax.hist(Y, bins=50, alpha=0.7, label="Targets")
ax.legend()
ax.set_title("Distribution of Predictions vs Targets")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
out_dir = os.path.dirname(__file__)
fig.savefig(
    os.path.join(out_dir, "hist_model_baseline.png"), dpi=150, bbox_inches="tight"
)
writer.add_figure("Histogram of Model prediction vs Baseline prediction", fig)
plt.close(fig)


"""
# Create matplotlib figure
M = 300
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_true[:M], label="Target", marker="o", markersize=3, linestyle="-", alpha=0.7)
ax.plot(
    y_pred[:M], label="Prediction", marker="x", markersize=3, linestyle="-", alpha=0.7
)
ax.set_title("Zoomed-in: First 300 Test Samples")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Value")
ax.legend()
fig.savefig(
    os.path.join(out_dir, "zoom_pred_vs_target.png"), dpi=150, bbox_inches="tight"
)
writer.add_figure("Prediction_vs_Target_RNNTest", fig)
plt.close(fig)
"""
#plot error line plot
mse = results["avg_mse"]
mare = results["avg_mare"]
baseline_mse = results["baseline_mse"]
baseline_mare = results["baseline_mare"]
num_samples = results["num_samples"]
mse_vals = [baseline_mse, mse]
mare_vals = [baseline_mare, mare]

print(f"Average MSE: {mse:.4f}, Average MARE: {mare:.4f}")
print(f"Baseline MSE: {baseline_mse:.4f}, Baseline MARE: {baseline_mare:.4f}")

# MSE plot comparaison
labels = ["Baseline", "Model"]
colors = ["mediumvioletred", "mediumaquamarine"] 
bar_width = 0.4
x_pos = [0,1]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(x_pos, mse_vals, width=bar_width, color=colors, edgecolor="black")

ax.set_ylabel("MSE", fontsize=12)

# X-ticks and labels
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=11)

# Annotate bars
for bar, val in zip(bars, mse_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.1, f"{val:.3f}",
            ha='center', va='bottom', fontsize=10)

ax.grid(True, axis='y', linestyle='--', alpha=0.4)
#plt.tight_layout()
ax.set_title("Mean Squared Error: Baseline vs Model", fontsize=14)
fig.savefig("bar_mse_comparison.png", dpi=150)
plt.close(fig)

# MARE plot comparaison
fig, ax = plt.subplots()
bars = ax.bar(x_pos, mare_vals, width=bar_width, color=colors, edgecolor="black")

# X-ticks and labels
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=11)

ax.set_ylabel("MARE", fontsize=12)
# Annotate bars
for bar, val in zip(bars, mare_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.1, f"{val:.3f}",
            ha='center', va='bottom', fontsize=10)

ax.grid(True, axis='y', linestyle='--', alpha=0.6)
ax.set_title("Mean Absolute Relative Error: Baseline vs Model")
fig.savefig("bar_mare_comparison.png", dpi=150)
plt.close(fig)


