import json
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "ca_results"

with open(OUT_DIR / "ca_errors.json") as f:
    results = json.load(f)

gru_errors = results["gru"]
baseline_errors = results["baseline"]
days = list(range(1, len(gru_errors) + 1))

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(days, gru_errors, label="GRU", color="steelblue", linewidth=2)
ax.plot(days, baseline_errors, label="Neighborhood average", color="coral", linewidth=2, linestyle="--")

ax.set_xlabel("Forecast horizon (days)")
ax.set_ylabel("MAE (°C)")
ax.set_title("CA autoregressive rollout — full horizon")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUT_DIR / "ca_mae_plot_full.png"
plt.savefig(output_path, dpi=150)
print(f"Saved to {output_path}")
plt.show()









exit(0)
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(days, gru_errors, label="GRU", color="steelblue", linewidth=2)
ax.plot(days, baseline_errors, label="Neighborhood average", color="coral", linewidth=2, linestyle="--")

ax.set_xlabel("Forecast horizon (days)")
ax.set_ylabel("MAE (°C)")
#ax.set_title("CA autoregressive rollout — MAE over forecast horizon")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = OUT_DIR / "ca_mae_plot.png"
plt.savefig(output_path, dpi=150)
print(f"Saved to {output_path}")
plt.show()