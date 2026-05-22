import torch
from torch.utils.data import TensorDataset, DataLoader
import os, sys
from pathlib import Path
from datetime import datetime
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.VanillaRNN import VanillaRNN
from src.models.lstm import LSTM
from src.models.gru import GRU
from src.utils.evaluate import evaluate_model, quick_test_sanity

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs" / "fixed_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# load test data
data = torch.load(DATA_DIR / "sst_test_set_fixed_47355.pt", map_location=device, weights_only=False)
X, Y = data["X"], data["Y"]
print(f"Test set: X={X.shape}, Y={Y.shape}")
test_loader = DataLoader(TensorDataset(X, Y), batch_size=256, shuffle=False)

# model configs
input_dim = 9
output_dim = 1
hidden_dim = 28
k = 5

MODELS = {
    "vanillarnn": {
        "class": VanillaRNN,
        "pattern": re.compile(r"^vanillarnn_gpu.*\.pt$"),
        "state_dict_key": "VanillaRNNStateDict",
    },
    "lstm": {
        "class": LSTM,
        "pattern": re.compile(r"^lstm_fixed_gpu.*\.pt$"),
        "state_dict_key": "LSTMStateDict",
    },
    "gru": {
        "class": GRU,
        "pattern": re.compile(r"^gru_fixed_gpu.*\.pt$"),
        "state_dict_key": "GRUStateDict",
    },
}

results_summary = {}

for model_name, config in MODELS.items():
    print(f"\n=== Testing {model_name.upper()} ===")

    # find most recent matching model file
    pt_files = sorted(
        [f for f in MODEL_DIR.iterdir() if config["pattern"].match(f.name)],
        key=lambda f: f.stat().st_mtime
    )
    if not pt_files:
        print(f"No model file found for {model_name}, skipping.")
        continue
    model_file = pt_files[-1]
    print(f"Loading: {model_file}")

    # instantiate and load
    model = config["class"](input_dim, hidden_dim, output_dim, num_layers=k)
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint[config["state_dict_key"]])

    # evaluate
    result_dico = evaluate_model(model, test_loader, device)

    # save per-model results
    torch.save(result_dico, OUT_DIR / f"test_{model_name}_result.pt")
    print(f"Saved {model_name} results")

    # print quantiles
    quick_test_sanity(result_dico["mse"], result_dico["mae"],
                      result_dico["absolute_error"], result_dico["squared_error"])

    # collect summary
    mse = result_dico["mse"].item()
    mae = result_dico["mae"].item()
    rmse = mse ** 0.5
    results_summary[model_name] = {"MSE": mse, "RMSE": rmse, "MAE": mae}

# print final table
print("\n=== FINAL RESULTS TABLE ===")
print(f"{'Model':<15} {'MSE':<12} {'RMSE':<12} {'MAE':<12}")
print("-" * 50)
for name, metrics in results_summary.items():
    print(f"{name:<15} {metrics['MSE']:<12.4f} {metrics['RMSE']:<12.4f} {metrics['MAE']:<12.4f}")
