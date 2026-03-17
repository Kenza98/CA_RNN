import torch
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

#loading error dicts for baseline

baseline = torch.load(OUT_DIR / "baselines.pt") #file, not dict... yet?
vanilla_rnn = torch.load(OUT_DIR / "testset_results.pt")
lstm = torch.load(OUT_DIR / "test_lstm_result.pt")
#gru= torch.load(OUT_DIR / "")

E = [baseline, vanilla_rnn, lstm]
#E = E.cpu().item()

for e in E:
    print(e.keys())

errs = [baseline['mse'],
        baseline['mae'],
        vanilla_rnn['mse_model_all'].mean().cpu().item(),
        vanilla_rnn['mare_model_all'].mean().cpu().item(),
        lstm['mse'],
         lstm['mae'] ]
labels = ["Baseline MSE = ",
          "Baseline MAE = ",
          "Vanilla MSE = ",
          "Vanilla MARE = ",
          "LSTM MSE = ",
          "LSTM MAE = "]

for l, e in zip(labels, errs):
    print(f"{l}{e}\n")