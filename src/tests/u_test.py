from scipy.stats import mannwhitneyu
import torch
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"


#loading error dicts for baseline

baseline = torch.load(OUT_DIR / "baselines.pt") #file, not dict... yet?
vanilla_rnn = torch.load(OUT_DIR / "testset_results.pt")
lstm = torch.load(OUT_DIR / "test_lstm_result.pt")

gru_fn = "test_gru_result.pt"   #GRU tests pt file name
gru= torch.load(OUT_DIR / gru_fn)   #GRU tests file path

#start with getting a numpy array from the baseline

baseline_err = baseline["absolute_errors"].cpu().numpy()
vanilla_err = vanilla_rnn["mare_model_all"].cpu().numpy()
lstm_err = lstm["absolute_error"].cpu().numpy()
gru_err = gru["absolute_error"].cpu().numpy()


mask = baseline_err > 0
baseline_err_clean = baseline_err[mask]
lstm_err_clean = lstm_err[mask]
gru_err_clean = gru_err[mask]
vanilla_err_clean = vanilla_err[mask]

print(f"Zeros in baseline: {(baseline_err_clean == 0).sum()}")
print(f"Zeros in lstm:     {(lstm_err_clean == 0).sum()}")
print(f"Zeros in gru:      {(gru_err_clean == 0).sum()}")
print(f"Zeros in vanilla:  {(vanilla_err_clean == 0).sum()}")

# Mann-Whitney U tests of each model vs baseline
#less    'two-sided'

ha='less'
stat_rnn, p_rnn = mannwhitneyu(vanilla_err_clean, baseline_err_clean, alternative=ha)
stat_lstm, p_lstm = mannwhitneyu(lstm_err_clean, baseline_err_clean, alternative=ha)
stat_gru, p_gru = mannwhitneyu(gru_err_clean, baseline_err_clean, alternative=ha)

print(f"Vanilla RNN vs Baseline: U={stat_rnn:.1f}, p={p_rnn:.4f}")
print(f"LSTM        vs Baseline: U={stat_lstm:.1f}, p={p_lstm:.4f}")
print(f"GRU         vs Baseline: U={stat_gru:.1f}, p={p_gru:.4f}")


'''
for err in [baseline_err, vanilla_err, lstm_err, gru_err]:
    print(type(err), err.shape)

print(f"Baseline mean: {baseline_err.mean():.4f}")
print(f"Vanilla mean:  {vanilla_err.mean():.4f}")
print(f"LSTM mean:     {lstm_err.mean():.4f}")
print(f"GRU mean:      {gru_err.mean():.4f}")

'''
