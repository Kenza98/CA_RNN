"""
Testing or retrieving info from the train, dev, and test sets of the SST dataset.
"""
def show_samples(fp):
    #this function takes the one file path of train, dev, or test datasets
    #and it shows the shape of X, Y in this file.
    data = torch.load(fp, map_location="cpu", weights_only=False)
    X, Y = data["X"], data["Y"]
    print(f"{fp} datafile loaded.")
    print(f"X: {X.shape}, Y: {Y.shape}")


import torch
from pathlib import Path
import torch.nn as nn
import torch.optim as optim


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs"

#load training dataset
fp_train = DATA_DIR / "sst_train_set_norm.pt"
show_samples(fp_train)


#load validation dataset
fp_val = DATA_DIR / "sst_val_set_norm.pt"
show_samples(fp_val)

#load test dataset
fp_test = DATA_DIR / "sst_test_set_norm.pt"
show_samples(fp_test)


ffp_train = DATA_DIR / "sst_train_set_acri_71427.pt"
ffp_test = DATA_DIR / "sst_test_set_acri_71427.pt"

#ffp_val = DATA_DIR / "sst_val_set_fixed_47355.pt"

for ffp in (ffp_train, ffp_test):
    show_samples(ffp)






