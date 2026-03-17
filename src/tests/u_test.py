from scipy.stats import mannwhitneyu
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"


#loading error dicts for each model

baseline = torch.load(OUT_DIR / "baselines.pt") #file, not dict... yet?
vanilla_rnn = torch.load(OUT_DIR / "testset_results.pt")
lstm = torch.load(OUT_DIR / "test_lstm_result.pt")
gru= torch.load(OUT_DIR / "")