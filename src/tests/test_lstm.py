import torch
from torch.utils.data import TensorDataset, DataLoader
import os, sys
from pathlib import Path
from datetime import datetime
import re
import argparse
# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
#imports that need project root
from src.models.lstm import LSTM
from src.utils.evaluate import evaluate_model, quick_test_sanity
####
#continue defining more paths...
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs" / "shuffle_True_ep100"
OUT_DIR.mkdir(parents=True, exist_ok=True)

#get the model file
parser = argparse.ArgumentParser()
parser.add_argument("--model-file", type=str, default=None,
                    help="Path to .pt file, e.g. models/lstm_gpu_44121.pt. If not provided, uses most recent lstm_gpu*.pt in MODEL_DIR.")
args = parser.parse_args()

if args.model_file is None:
    # fall back to most recent lstm_gpu*.pt
    pattern = re.compile(r"^lstm_gpu.*\.pt$")
    pt_files = sorted(
        [f for f in MODEL_DIR.iterdir() if pattern.match(f.name)],
        key=lambda f: f.stat().st_mtime
    )
    if not pt_files:
        raise FileNotFoundError(f"No matching lstm_gpu*.pt files found in {MODEL_DIR}")
    model_file = pt_files[-1]
else:
    model_file = PROJECT_ROOT / args.model_file

print(model_file)

# check if cuda devise available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)

# LOAD DATA FILE, GET BASELINE

test_data_file = DATA_DIR / "sst_test_set.pt"
test_data = torch.load(test_data_file, map_location="cpu", weights_only=False)
X = test_data["X"]
Y = test_data["Y"]
total_samples = Y.shape[0]
test_dataset = TensorDataset(X, Y)

# keep data unshuffled for reproducibility.
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


timestamp = os.path.getmtime(model_file)
print("Last model update date: ", datetime.fromtimestamp(timestamp))

# load the model checkpoint on disk (cpu)
output_dim = 1
input_dim = 9
hidden_dim =56
k=4
checkpoint = torch.load(model_file, map_location=device)

print("Top-level keys in model file:", checkpoint.keys(), flush=True)

model = LSTM(input_dim, hidden_dim, output_dim, k)
model.load_state_dict(checkpoint["LSTMStateDict"])

# GETTING RESULTS WITH HELPER FCT
my_dict = evaluate_model(model, test_loader, device)

# SAVING RESULTS TO PT FILE
output_file = OUT_DIR / "test_lstm_result.pt"
torch.save(my_dict, output_file)
model_class = model.__class__.__name__
print(f"Saved {model_class} evaluation")


# ***PRINT SOME QUANTILES***
mse = my_dict["mse"]
mae= my_dict["mae"]
ae_tensor = my_dict["absolute_error"]
se_tensor = my_dict["squared_error"]
quick_test_sanity(mse, mae, ae_tensor, se_tensor)
