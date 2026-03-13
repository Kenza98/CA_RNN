import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os, sys
from pathlib import Path
from datetime import datetime

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
print(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

from src.models.lstm import LSTM
from src.models.VanillaRNN import VanillaRNN
from src.utils.evaluate import evaluate_model, get_baseline

# check if file was ran with --use-gpu + if cuda devise available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)

# LOAD DATA FILE, GET BASELINE

test_data_file = (
    DATA_DIR / "sst_test_set.pt"
)  # \\CHECK all data test + train in this dir?

# load data on cpu
test_data = torch.load(test_data_file, map_location="cpu")
X = test_data["X"]
Y = test_data["Y"]
total_samples = Y.shape[0]
test_dataset = TensorDataset(X, Y)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

# LOAD MODEL FILE, GET MODELS TEST RESULTS
model_file = MODEL_DIR / "lstm_moore.pt"
output_file = OUT_DIR / "test_lstm_result.pt"


# load the model checkpoint on disk (cpu)
output_dim = 1
input_dim = 9
hidden_dim = 7 * 8
checkpoint = torch.load(model_file, map_location="cpu")
model = LSTM(input_dim, hidden_dim, output_dim)
model.load_state_dict(checkpoint["lstmMoore_stateDict"])
model = model.to(device)

# === SAVING RESULTS TO PT FILE ===
my_dict = evaluate_model(model, test_loader, device)
torch.save(my_dict, output_file)
print("Saved {model.name()} evaluation")


exit(0)

# little check that i have the recently trained model
# timestamp = os.path.getmtime(model_file)
# print("Last model update:", datetime.fromtimestamp(timestamp))

# === SHOWING QUANTILES ===
quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
print(ae_tensor.shape)

print(f"flattened shape : {ae_flat.shape}\n")


ae_q = torch.quantile(ae_flat, quantiles)
se_q = torch.quantile(se_flat, quantiles)

print("\n=== Absolute Error Quantiles ===")
for q, v in zip(quantiles, ae_q):
    print(f"{q.item():>5.2f} : {v.item():.6f}")

print("\n=== Squared Error Quantiles ===")
for q, v in zip(quantiles, se_q):
    print(f"{q.item():>5.2f} : {v.item():.6f}")

# ONLY QIUARTILES:

quartiles = torch.tensor([0.25, 0.5, 0.75])
print(torch.quantile(ae_tensor.flatten(), quartiles))
print("RMSE: ", torch.sqrt(mse).item())
