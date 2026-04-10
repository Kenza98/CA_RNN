import torch
from torch.utils.data import TensorDataset, DataLoader
import os,sys
from pathlib import Path
from datetime import datetime
# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.utils.evaluate import get_baseline, quick_test_sanity
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs" / "shuffle_True_ep100"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# check if cuda devise available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)

baseline_fp = OUT_DIR / "baselines.pt" #baseline pt file path

#LOAD TEST DATA FILE
data_file = DATA_DIR / "sst_test_set.pt"

# load data on cpu
test_data = torch.load(data_file, map_location="cpu", weights_only=False)
X = test_data["X"]
Y = test_data["Y"]
total_samples = Y.shape[0]
print(f"Total samples: {total_samples}\n")
test_dataset = TensorDataset(X, Y)

#keep data unshuffled (useless)
data_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# === SAVING RESULTS TO PT FILE ===
start = datetime.now()
my_dict = get_baseline(data_loader, device)
print(f"Baseline computed in {datetime.now() - start}")

quick_test_sanity(mse, mae, ae_tensor, se_tensor)

torch.save(my_dict, baseline_fp)
print("Saved baseline evaluation.", end="\n")
