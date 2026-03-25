import torch
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "shuffle_True_ep100"

def find_result_file(pattern_str):
    pattern = re.compile(pattern_str)
    files = [f for f in OUT_DIR.iterdir() if pattern.match(f.name)]
    if not files:
        raise FileNotFoundError(f"No file matching '{pattern_str}' in {OUT_DIR}")
    return sorted(files, key=lambda f: f.stat().st_mtime)[-1]


#loading error dicts for baseline
bf=find_result_file(r".*baseline.*\.pt$") #baseline file
lf=find_result_file(r".*lstm.*\.pt$") #baseline file
gf=find_result_file(r".*gru.*\.pt$") #baseline file
vf=find_result_file(r".*vrnn.*\.pt$") #baseline file

print(f"The files :\n1. {bf}\n2. {lf}\n3. {gf}\n4. {vf}\n\n-----------------\n\n")
exit(0)
baseline     = torch.load(bf, weights_only=False)
vanilla_rnn  = torch.load(vf, weights_only=False)
lstm         = torch.load(lf, weights_only=False)
gru          = torch.load(gf, weights_only=False)



# loading error dicts for each model
baseline = torch.load(OUT_DIR / "baselines.pt")  # file, not dict... yet?
vanilla_rnn = torch.load(OUT_DIR / "testset_results.pt")
lstm = torch.load(OUT_DIR / "test_lstm_result.pt")
gru_fn = "test_gru_result.pt"  # GRU tests pt file name
gru = torch.load(OUT_DIR / gru_fn)  # GRU tests file path

E = [baseline, vanilla_rnn, lstm, gru]

errs = [
    baseline["mse"],
    baseline["mae"],
    torch.mean(vanilla_rnn["mse_model_all"]).cpu().item(),
    torch.mean(vanilla_rnn["mare_model_all"]).cpu().item(),
    lstm["mse"],
    lstm["mae"],
    gru["mse"],
    gru["mae"],
]

labels = [
    "Baseline MSE = ",
    "Baseline MAE = ",
    "Vanilla MSE = ",
    "Vanilla MARE = ",
    "LSTM MSE = ",
    "LSTM MAE = ",
    "GRU MSE =  ",
    "GRU MAE = ",
]

for l, e in zip(labels, errs):
    print(f"{l}{e}\n")



'''
for e in E:
    print(f"{e.keys()}\n")
'''
