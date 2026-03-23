from scipy.stats import mannwhitneyu
import torch
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"


#loading error dicts for baseline
baseline = torch.load(OUT_DIR / "baselines.pt")
vanilla_rnn = torch.load(OUT_DIR / "test_vrnn_result.pt")
lstm = torch.load(OUT_DIR / "test_lstm_result.pt")
gru= torch.load(OUT_DIR / "test_gru_result.pt")   


#start with getting a numpy array from the baseline
baseline_err = baseline["absolute_error"].cpu().numpy()
vanilla_err = vanilla_rnn["absolute_error"].cpu().numpy()
lstm_err = lstm["absolute_error"].cpu().numpy()
gru_err = gru["absolute_error"].cpu().numpy()

print(vanilla_err.mean(), baseline_err.mean())

"""
mask = baseline_err > 0
baseline_err_clean = baseline_err[mask]
lstm_err_clean = lstm_err[mask]
gru_err_clean = gru_err[mask]
vanilla_err_clean = vanilla_err[mask]


print(f"Zeros in baseline: {(baseline_err == 0).sum()}")
print(f"Zeros in lstm:     {(lstm_err == 0).sum()}")
print(f"Zeros in gru:      {(gru_err == 0).sum()}")
print(f"Zeros in vanilla:  {(vanilla_err == 0).sum()}")
"""

# Mann-Whitney U tests of each model vs baseline
#less    'two-sided'
print("testing Vanilla RNN != baseline")
stat_rnn, p_rnn = mannwhitneyu(vanilla_err, baseline_err, alternative='two-sided')
print(f"Vanilla RNN vs Baseline: U={stat_rnn:.1f}, p={p_rnn:.4f}")

ha='less'
stat_rnn, p_rnn = mannwhitneyu(vanilla_err, baseline_err, alternative=ha)
stat_lstm, p_lstm = mannwhitneyu(lstm_err, baseline_err, alternative=ha)
stat_gru, p_gru = mannwhitneyu(gru_err, baseline_err, alternative=ha)

print(f"Vanilla RNN vs Baseline: U={stat_rnn:.1f}, p={p_rnn:.4f}")
print(f"LSTM        vs Baseline: U={stat_lstm:.1f}, p={p_lstm:.4f}")
print(f"GRU         vs Baseline: U={stat_gru:.1f}, p={p_gru:.4f}")

