import torch
from torch.utils.data import random_split
import numpy as np
import os
from SlidingWindowDs import *
import xarray as xr
import gc


# preprocessing pre-tensor
def Dataset_to_pt(ds, output_file):
    N = len(ds)
    print(N)
    per_nan = 0
    per_out = 0
    X, Y = [], []
    for i in range(len(ds)):
        try:
            x, y = ds[i]
            if torch.isnan(x).any() or torch.isnan(y).any():
                per_nan += 1
                continue
            X.append(x)
            Y.append(y)

        except IndexError:
            # print(f"Index {i} out of bounds for dataset of length {len(ds)}")
            per_out += 1
            continue

        if (i % 10000) in range(10):
            print(f"step {i} ... {N - i} samples remaining")
        if i % 100000 in range(10):
            print(f"Processing data {int(i*100/N)}% complete")
            print(f"percentage of nan : {int(per_nan * 100 / N)} %")
            print(f"percentage of out of bound {int(per_out * 100 / N)}%")

    if not X:
        print(f"No valid samples for {output_file}")
        return

    X_tensor = torch.stack(X)
    Y_tensor = torch.stack(Y)
    print(f"X shape: {X_tensor.shape}, Y shape: {Y_tensor.shape}")
    dir_name = os.path.dirname(output_file)  # je récup le nom du dossier
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)  # only creates new dir if it doesn't exist
        # exists_ok is True by default in Python 3.2+
    # create .pt file
    try:
        torch.save(
            {
                "X": X_tensor,
                "Y": Y_tensor,
            },
            output_file,
        )
    except FileNotFoundError:
        print("File not found. Please check the output .pt file path.")
    finally:
        print(f"Dataset saved to {output_file}")


# use Dask to load data in chunks of 100 time steps.
dataset = xr.open_dataset("train_sst.nc", chunks={"time": 100})
sst_train = dataset["thetao"]
dataset.close()
if "depth" in sst_train.dims:
    sst_train = sst_train.isel(depth=0)


train_ds = SlidingWindowDs(sst_train, seq_length=8)


Dataset_to_pt(train_ds, "training_set.pt")
del sst_train, train_ds, dataset
gc.collect()

dataset = xr.open_dataset("test_sst.nc", chunks={"time": 100})

sst_test = dataset["thetao"]
dataset.close()
if "depth" in sst_test.dims:
    sst_test = sst_test.isel(depth=0)


del dataset
gc.collect()

test_ds = SlidingWindowDs(sst_test, seq_length=8)
output_file = "sst_test_set.pt"

Dataset_to_pt(test_ds, output_file)

del sst_test
gc.collect()
