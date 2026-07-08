"""
Testing or retrieving info from the train, dev, and test sets of the SST dataset.
"""

import torch
from pathlib import Path
import torch.nn as nn
import torch.optim as optim


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

#load training dataset
data = torch.load(DATA_DIR / "sst_train_set_norm.pt", map_location="cpu", weights_only=False)
X, Y = data["X"], data["Y"]

#load validation dataset
val_data = torch.load(DATA_DIR / "sst_val_set_norm.pt", map_location="cpu", weights_only=False)
X_val, Y_val = val_data["X"], val_data["Y"]


#load test dataset
test_data = torch.load(DATA_DIR / "sst_test_set_norm.pt", map_location="cpu", weights_only=False)
X_test, Y_test = test_data["X"], test_data["Y"]

print(f"Testing set: {X_test.shape}, {Y_test.shape}")
