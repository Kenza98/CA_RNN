import torch
from torch.utils.data import TensorDataset, DataLoader
import os,sys
from pathlib import Path
from datetime import datetime
from src.utils.evaluate import get_baseline, quick_test_sanity

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
print(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"


# check if file was ran with --use-gpu + if cuda devise available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)

# LOAD BASELINE FILE
baseline_file = OUT_DIR / "baselines.pt"


#LOAD DATA FILE
data_file = DATA_DIR / "sst_test_set.pt"

# load data on cpu
test_data = torch.load(data_file, map_location="cpu")
X = test_data["X"]
Y = test_data["Y"]
total_samples = Y.shape[0]
#print(f"Total samples: {total_samples}\n")
#exit(0)
test_dataset = TensorDataset(X, Y)
#keep data unshuffled for reproducibility.
data_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# === SAVING RESULTS TO PT FILE ===
my_dict = get_baseline(data_loader, device)
#

# ***PRINT SOME QUANTILES***
mse = my_dict["mse"]
mae= my_dict["mae"]
ae_tensor = my_dict["absolute_error"]
se_tensor = my_dict["squared_error"]
quick_test_sanity(mse, mae, ae_tensor, se_tensor)


for k, v in my_dict.items():
    print(f"{k}: {type(v)}, {v.shape if hasattr(v, 'shape') else v}")

torch.save(my_dict, baseline_file)
print("Saved baseline evaluation.", end="\n")
