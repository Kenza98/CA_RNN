import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

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