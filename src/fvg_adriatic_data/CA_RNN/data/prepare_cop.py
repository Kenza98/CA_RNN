import torch
from torch.utils.data import random_split
import numpy as np
import os
from SlidingWindowDs import *
import xarray as xr

# Convert an xarray dataset to a PyTorch tensor.

dataset = xr.open_dataset("data/subset_adriatic_sst.nc")
time = dataset.time
surface_temp = dataset["thetao"]

"""
print("Latitude range:", surface_temp.latitude.min().item(), "to", surface_temp.latitude.max().item())
print("Longitude range:", surface_temp.longitude.min().item(), "to", surface_temp.longitude.max().item())
"""
total_elements = surface_temp.size
num_nans = np.isnan(surface_temp.values).sum()
percent_nans = (num_nans / total_elements) * 100
print(f"Percentage of NaNs in the entire dataset: {percent_nans:.2f}%")

#exit()
# Quick summary stats from the raw xarray DataArray
print("Min value:", surface_temp.min().item())
print("Max value:", surface_temp.max().item())
print("Mean value:", surface_temp.mean().item())
print("Number of NaNs:", np.isnan(surface_temp.values).sum())


tensor_sst = torch.tensor(surface_temp.values, dtype=torch.float32)
tensor_sst = tensor_sst.squeeze(dim=1)

# print(tensor_sst.shape)
# depth = surface_temp.depth.values
# min_depth_index = np.argmin(depth)

# tensor_sst = tensor_temp[:, min_depth_index, :, :]

tensor_sst = tensor_sst.unsqueeze(-1)  # Add feature dimension at the end

try:
    output_file = "./data/cop_ml_ready.pt"  # pt file name
except FileNotFoundError:
    print("File not found.")


full_dataset = SlidingWindowDs(tensor_sst, 8)  # seq_length = 8

N = len(full_dataset)
print(N)
train_size = int(0.8 * N)
test_size = N - train_size
lengths = [train_size, test_size]
train_dataset, test_dataset = random_split(full_dataset, lengths)

# extract x,y pairs
X_TRAIN, Y_TRAIN = zip(*[train_dataset[i] for i in range(len(train_dataset))])
X_TEST, Y_TEST = zip(*[test_dataset[i] for i in range(len(test_dataset))])

X_TEST = torch.stack(X_TEST)
Y_TEST = torch.stack(Y_TEST)
X_TRAIN = torch.stack(X_TRAIN)
Y_TRAIN = torch.stack(Y_TRAIN)

print(X_TRAIN.shape, Y_TRAIN.shape, X_TEST.shape, Y_TEST.shape)

# save to EXISTING .pt file
try:
    pt_file = torch.load(output_file)
    print("file loaded successfully")
    pt_file["X_TRAIN"] = X_TRAIN
    pt_file["Y_TRAIN"] = Y_TRAIN
    pt_file["X_TEST"] = X_TEST
    pt_file["Y_TEST"] = Y_TEST
    torch.save(pt_file, output_file)

except FileNotFoundError:
    """
    CASE FIRST TIME CREATING PT FILE
    """
    print("Creating new .pt file ... \n ...\n")
    data_to_save = {
        "X_TRAIN": X_TRAIN,
        "Y_TRAIN": Y_TRAIN,
        "X_TEST": X_TEST,
        "Y_TEST": Y_TEST,
    }

    torch.save(data_to_save, output_file)
print(
    f"Created sliding window dataset with train size {len(train_dataset)} and test size {len(test_dataset)} samples."
)
