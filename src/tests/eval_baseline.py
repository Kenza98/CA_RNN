from src.utils.evaluate import evaluate_model, get_baseline
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader
import sys

#check if file was ran with --use-gpu + if cuda devise available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
"""
sys.path.append(str(PROJECT_ROOT))
print(PROJECT_ROOT)
"""

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"

test_data_file = DATA_DIR / "sst_test_set.pt"
#load data on cpu
test_data = torch.load(test_data_file, map_location="cpu")
X = test_data['X']
Y = test_data['Y']
dataset = TensorDataset(X, Y)
data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Save Baseline results
output_file = OUT_DIR / "baselines.pt"
baseline_result = get_baseline(data_loader, device)
torch.save(baseline_result, output_file)
print("Saved baseline evaluation")