import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-gpu", action="store_true", help="Use GPU for tensor stacking"
)
args = parser.parse_args()
use_gpu = args.use_gpu
import torch
import os
import xarray as xr
import gc
import sys
import numpy as np

sys.stdout.flush()
from tqdm import tqdm

import torch
import xarray as xr
import numpy as np
import gc
from tqdm import tqdm


def build_learning_set_from_xarray(
    nc_path, seq_length=8, chunk_size=200, use_gpu=False
):
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}", flush=True)

    # Open with lazy loading
    ds = xr.open_dataset(nc_path, chunks={"time": chunk_size})
    sst = ds["thetao"]
    if "depth" in sst.dims:
        sst = sst.isel(depth=0)

    # Precompute global mean
    global_mean = float(sst.mean().compute())
    print(f"Global mean SST: {global_mean:.4f}", flush=True)

    X_list, Y_list = [], []

    total_time = sst.sizes["time"]
    # Iterate over big time chunks
    for start in tqdm(
        range(0, total_time - seq_length, chunk_size), desc="Processing chunks"
    ):
        end = min(total_time, start + chunk_size + seq_length)
        block = (
            sst.isel(time=slice(start, end)).compute().values.astype(np.float32)
        )  # shape: (T, H, W)

        # Loop over valid time indices in this block
        for t in range(0, block.shape[0] - seq_length):
            seq_block = block[t : t + seq_length]
            target_map = block[t + seq_length]

            # Loop over spatial grid except borders
            for i in range(1, block.shape[1] - 1):
                for j in range(1, block.shape[2] - 1):
                    neigh = seq_block[:, i - 1 : i + 2, j - 1 : j + 2]

                    if np.isnan(neigh).any() or np.isnan(target_map[i, j]):
                        neigh = np.nan_to_num(neigh, nan=global_mean)
                        target_val = global_mean
                    else:
                        target_val = target_map[i, j]

                    X_list.append(neigh.reshape(seq_length, -1))
                    Y_list.append([target_val])

                    # After creating neigh and target_val inside the loop:
                    if len(X_list) < 5:  # first 5 samples
                        print("Sample", len(X_list))
                        print("Neighborhood:\n", neigh)
                        print("Target:", target_val, "\n")

                    if (
                        np.isnan(seq_block).any() and len(X_list) < 10
                    ):  # print a few NaN cases
                        print("NaN replaced sample", len(X_list))
                        print("Neighborhood after NaN replace:\n", neigh)
                        print("Target after NaN replace:", target_val, "\n")

        gc.collect()

    # Stack into tensors & send to GPU if needed
    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(np.array(Y_list), dtype=torch.float32).to(device)

    ds.close()
    return X_tensor, Y_tensor


X_train, Y_train = build_learning_set_from_xarray(
    "train_sst.nc",
    seq_length=8,
    chunk_size=200,
    use_gpu=True,
)

print(f"X shape: {X_train.shape}, Y shape: {Y_train.shape}")


torch.save({"X": X_train, "Y": Y_train}, "sst_train_set.pt")
