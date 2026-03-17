from scipy.stats import mannwhitneyu
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"


#loading error dicts for baseline

baseline = torch.load(OUT_DIR / "baselines.pt") #file, not dict... yet?
vanilla_rnn = torch.load(OUT_DIR / "testset_results.pt")
lstm = torch.load(OUT_DIR / "test_lstm_result.pt")
#gru= torch.load(OUT_DIR / "")

#start with getting a numpy array from the baseline

baseline_err = baseline["babsolute_errors"].cpu().numpy()
vanilla_err = vanilla_rnn['mare_model_all'].cpu().numpy()
lstm_err = lstm['labsolute_error'].cpu().numpy()

# Mann-Whitney U tests of each model vs baseline
stat_rnn, p_rnn = mannwhitneyu(vanilla_err, baseline_err, alternative='two-sided')
stat_lstm, p_lstm = mannwhitneyu(lstm_err, baseline_err, alternative='two-sided')

print(f"Vanilla RNN vs Baseline: U={stat_rnn:.1f}, p={p_rnn:.4f}")
print(f"LSTM        vs Baseline: U={stat_lstm:.1f}, p={p_lstm:.4f}")




'''
for err in [baseline_err, vanilla_err, lstm_err]:
    print(type(err), err.shape)
'''



