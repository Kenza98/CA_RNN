from scipy.stats import mannwhitneyu
import torch
from pathlib import Path
import numpy as np
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "shuffle_True_ep100"

def find_result_file(pattern_str):
    pattern = re.compile(pattern_str)
    files = [f for f in OUT_DIR.iterdir() if pattern.match(f.name)]
    if not files:
        raise FileNotFoundError(f"No file matching '{pattern_str}' in {OUT_DIR}")
    return sorted(files, key=lambda f: f.stat().st_mtime)[-1]

def show_quartiles(err_tensor):
    # computes quartiles and returns them + shows them
    quartiles = torch.tensor([0.25, 0.5, 0.75])
    print(torch.quantile(err_tensor.flatten(), quartiles))
    for v in quartiles:
        print(f"{v.item():.6f}")



#loading error dicts for baseline
bf=find_result_file(r".*baseline.*\.pt$") #baseline file
lf=find_result_file(r".*lstm.*\.pt$") #baseline file
gf=find_result_file(r".*gru.*\.pt$") #baseline file
vf=find_result_file(r".*vrnn.*\.pt$") #baseline file

baseline     = torch.load(bf, weights_only=False)
vanilla_rnn  = torch.load(vf, weights_only=False)
lstm         = torch.load(lf, weights_only=False)
gru          = torch.load(gf, weights_only=False)

L = bf, vf, lf, gf
print(f"Successfully loaded all file : {L}\n")

#start with getting a numpy array from the baseline
baseline_err = baseline["absolute_error"].cpu().numpy()
vanilla_err = vanilla_rnn["absolute_error"].cpu().numpy()
lstm_err = lstm["absolute_error"].cpu().numpy()
gru_err = gru["absolute_error"].cpu().numpy()

#showing quartiles

Err = baseline_err, vanilla_err, lstm_err, gru_err

for e in Err:
    show_quartiles(e)

exit(0)





"""
mask = baseline_err > 0
baseline_err_clean = baseline_err[mask]
lstm_err_clean = lstm_err[mask]
gru_err_clean = gru_err[mask]
vanilla_err_clean = vanilla_err[mask]

#print perfect predictions count
print(f"Zeros in baseline: {(baseline_err == 0).sum()}")
print(f"Zeros in lstm:     {(lstm_err == 0).sum()}")
print(f"Zeros in gru:      {(gru_err == 0).sum()}")
print(f"Zeros in vanilla:  {(vanilla_err == 0).sum()}")
"""
# Mann-Whitney U tests of each model vs baseline
#less    'two-sided'
#print("testing Vanilla RNN != baseline")
#stat_rnn, p_rnn = mannwhitneyu(vanilla_err, baseline_err, alternative='two-sided')
#print(f"Vanilla RNN vs Baseline: U={stat_rnn:.1f}, p={p_rnn:.4f}")

ha='less'
stat_rnn, p_rnn = mannwhitneyu(vanilla_err, baseline_err, alternative=ha)
stat_lstm, p_lstm = mannwhitneyu(lstm_err, baseline_err, alternative=ha)
stat_gru, p_gru = mannwhitneyu(gru_err, baseline_err, alternative=ha)

print(f"Vanilla RNN vs Baseline: U={stat_rnn:.1f}, p={p_rnn:.4f}")
print(f"LSTM        vs Baseline: U={stat_lstm:.1f}, p={p_lstm:.4f}")
print(f"GRU         vs Baseline: U={stat_gru:.1f}, p={p_gru:.4f}")

