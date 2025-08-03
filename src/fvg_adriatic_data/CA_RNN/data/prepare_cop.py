import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use-gpu", action="store_true", help="Use GPU for tensor stacking")
args = parser.parse_args()

use_gpu = args.use_gpu
import torch
import os
from SlidingWindowDs import *
import xarray as xr
import gc
from torch.utils.data import DataLoader
import sys
sys.stdout.flush()
# preprocessing with batches
def Dataset_to_pt(ds, output_file, batch_size=256, num_workers=8, use_gpu=False):
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
    	print("GPU name:", torch.cuda.get_device_name(0), flush=True)
        print("CUDA version:", torch.version.cuda, flush=True)
    else:
    	print("No GPU detected. Running on CPU.", flush=True)
    N = len(ds)
    print(N)
    per_nan = 0
    per_out = 0
    X, Y = [], []
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        # pin_memory=True,
        # drop_last=True,
        shuffle=False,  # No need to shuffle for saving to .pt
        collate_fn=lambda b: b,  # keep raw (x, y) tuples
    )
    print(f"DataLoader created with {len(loader)} batches of size {batch_size}")
    processed = 0
    for batch in loader:
        print("new batch!")
        for sample in batch:
            try:
                x, y = sample
                if torch.isnan(x).any() or torch.isnan(y).any():
                    per_nan += 1
                    continue
                X.append(x)
                Y.append(y)
            except IndexError:
                # print(f"Index {i} out of bounds for dataset of length {len(ds)}")
                per_out += 1
                continue
        processed += len(batch)
        if (processed % 10000) < batch_size:
            print(f"{processed}/{N} processed | NaN: {per_nan} | Out: {per_out}")

    if not X:
        print(f"No valid samples for {output_file}")
        return

    X_tensor = torch.stack(X).to(device)
    Y_tensor = torch.stack(Y).to(device)

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

Dataset_to_pt(train_ds, "training_set.pt", batch_size=256, num_workers=8, use_gpu=use_gpu)
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

Dataset_to_pt(test_ds, output_file, batch_size=256, num_workers=8, use_gpu=use_gpu)

del sst_test
gc.collect()
