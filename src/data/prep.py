"""Shared data prep pipeline for CA_RNN.

Loads Copernicus SST, builds (X, Y) tensors via a selected extractor, and
saves one .pt per split. Extraction logic lives in extractors.py.

Usage:
    python -m src.data.prep --experiment 1 --features nca
"""

import argparse
import os

# from datetime import date
from pathlib import Path

# import copernicusmarine
import torch
import xarray as xr
from tqdm import tqdm

from .extractors import neighborhood_valid_mask, EXTRACTORS
from .download import BBOX, DATASET_ID, SPLITS, raw_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment",
        type=int,
        required=True,
        help="Experiment number, used in the output filename",
    )
    p.add_argument(
        "--features",
        choices=sorted(EXTRACTORS),
        required=True,
        help="nn = point-wise history, nca = 3x3 neighborhood",
    )
    p.add_argument("--seq-length", type=int, default=6)
    p.add_argument("--chunk-size", type=int, default=200)

    return p.parse_args()


def load_dataset(split_name, chunk_size=200):
    """Open the local netCDF for a split, select the surface level."""
    ds = xr.open_dataset(raw_path(split_name)).chunk({"time": chunk_size})
    sst = ds["thetao"]
    if "depth" in sst.dims:
        sst = sst.isel(depth=0)
        #print(sst.dtype)
    return ds, sst 


def build_learning_set(sst, extractor, seq_length=6, chunk_size=200):
    # only showing stats of the data
    # single pass over dask of the data by calling .compute() only once
    stats = xr.Dataset(
        {"mean": sst.mean(skipna=True), "std": sst.std(skipna=True)}
    ).compute()
    global_mean = float(stats["mean"])
    global_std = float(stats["std"])
    print(f"Global mean= {global_mean:.4f}", flush=True)
    print(f"Global std= {global_std:.4f}", flush=True)

    # initialize the objects this fct builds
    X_chunks, Y_chunks = [], []
    total_time = sst.sizes["time"]  # nb of timesteps in dataset

    for start in tqdm(
        range(0, total_time - seq_length, chunk_size), desc="Processing chunks"
    ):
        # chunks overlap by seq_length so each window has its target;
        # clamp at total_time for the last chunk
        end = min(total_time, start + chunk_size + seq_length)

        block_lazy = sst.isel(
            time=slice(start, end)
        )  # lazy loads the data array sliced
        block_xr = block_lazy.compute()  # DataArray wrapping a dask array
        block_np = block_xr.values  # DataArray wrapping a numpy array
        block = torch.from_numpy(block_np).float()  # finally, torch from the np

        for t in range(block.shape[0] - seq_length):
            seq_block = block[t : t + seq_length]
            target_map = block[t + seq_length]

            X_t, Y_t = extractor(seq_block, target_map)

            # same criterion for both arms: full 3x3 neighborhood must be finite
            valid_mask = neighborhood_valid_mask(seq_block, target_map)

            X_t = X_t[valid_mask]
            Y_t = Y_t[valid_mask]

            assert X_t.shape[0] == Y_t.shape[0], "X/Y cell count mismatch"
            assert not torch.isnan(X_t).any(), "NaNs remain in X_t after masking"
            assert not torch.isnan(Y_t).any(), "NaNs remain in Y_t after masking"

            X_chunks.append(X_t)
            Y_chunks.append(Y_t)

        del block_xr, block

    return torch.cat(X_chunks), torch.cat(Y_chunks), global_mean, global_std


def main():
    args = parse_args()
    extractor = EXTRACTORS[args.features]
    slurm_job_id = os.environ.get(
        "SLURM_JOB_ID", "local"
    )  # Doc: os.environ.get(key, default)

    for split_name, (sd, ed) in SPLITS.items():
        print(f"\n------ {split_name}: {sd} -> {ed} ------", flush=True)
        ds, sst = load_dataset(split_name)
        X, Y, global_mean, global_std = build_learning_set(sst, extractor)
        print(f"X: {tuple(X.shape)}  Y: {tuple(Y.shape)}", flush=True)

        out = OUT_DIR / f"{args.experiment}_{args.features}_{split_name}.pt"
        torch.save(
            {
                "X": X,
                "Y": Y,
                "config": {
                    "experiment": args.experiment,
                    "features": args.features,
                    "seq_length": args.seq_length,
                    "split": split_name,
                    "start_time": sd,
                    "end_time": ed,
                    "dataset_id": DATASET_ID,
                    "bbox": BBOX,
                    "global_mean": global_mean,
                    "global_std": global_std,
                    "slurm_job_id": slurm_job_id,
                },
            },
            out,
        )
        print(f"Saved: {out}", flush=True)

        ds.close()
        del X, Y


if __name__ == "__main__":
    main()
