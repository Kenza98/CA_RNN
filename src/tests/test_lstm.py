import torch
from torch.utils.data import TensorDataset, DataLoader
import os, sys
from pathlib import Path
from datetime import datetime
from src.models.lstm import LSTM
from src.models.VanillaRNN import VanillaRNN
from src.utils.evaluate import evaluate_model, quick_test_sanity
import re

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
print(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs" / "shuffle_True_ep100"


# check if file was ran with --use-gpu + if cuda devise available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)

# LOAD DATA FILE, GET BASELINE

test_data_file = DATA_DIR / "sst_test_set.pt"
test_data = torch.load(test_data_file, map_location="cpu")
X = test_data["X"]
Y = test_data["Y"]
total_samples = Y.shape[0]
test_dataset = TensorDataset(X, Y)

# keep data unshuffled for reproducibility.
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# LOAD MODEL FILE, GET MODELS TEST RESULTS
# ask user which model file to use
print("Enter 0 to use the most recent model file, or enter the model filename (e.g. lstm_gpu_44121.pt):")
user_input = input("> ").strip()

if user_input == "0":
    pattern = re.compile(r"^lstm_gpu.*\.pt$") # r " " is raw string
    pt_files = sorted(
        [f for f in MODEL_DIR.iterdir() if pattern.match(f.name)],
        key=os.path.getmtime
    )
    if not pt_files:
        raise FileNotFoundError(f"No matching lstm_gpu*.pt files found in {MODEL_DIR}")
    model_file = pt_files[-1]
    print(f"Using most recent model: {model_file.name}")
else:
    model_file = MODEL_DIR / user_input
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    print(f"Using model: {model_file.name}")


#model_file = MODEL_DIR / "lstm_gpu_44121.pt"
timestamp = os.path.getmtime(model_file)
print("Last model update date: ", datetime.fromtimestamp(timestamp))



# load the model checkpoint on disk (cpu)
output_dim = 1
input_dim = 9
hidden_dim = 7 * 8
checkpoint = torch.load(model_file, map_location="cpu")

print("Top-level keys in model file:", checkpoint.keys(), flush=True)

model = LSTM(input_dim, hidden_dim, output_dim)
model.load_state_dict(checkpoint["LSTMStateDict"])

# GETTING RESULTS WITH HELPER FCT
my_dict = evaluate_model(model, test_loader, device)

# SAVING RESULTS TO PT FILE
output_file = OUT_DIR / "test_lstm_result.pt"
torch.save(my_dict, output_file)
print(f"Saved {model.name()} evaluation")


# ***PRINT SOME QUANTILES***
mse = my_dict["mse"]
mae= my_dict["mae"]
ae_tensor = my_dict["absolute_error"]
se_tensor = my_dict["squared_error"]
quick_test_sanity(mse, mae, ae_tensor, se_tensor)
