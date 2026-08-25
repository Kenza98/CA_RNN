"""Shared data prep pipeline for CA_RNN.

Loads Copernicus SST, builds (X, Y) tensors via a selected extractor, and
saves one .pt per split. Extraction logic lives in extractors.py.

Usage:
    python -m src.data.prep --experiment 1 --features nca
"""

import argparse
import os
from datetime import date
from pathlib import Path

import copernicusmarine
import torch
import xarray as xr
from tqdm import tqdm

from .extractors import EXTRACTORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "cmems_mod_med_phy-temp_my_4.2km_P1D-m"
BBOX = dict(
    minimum_longitude=12,
    maximum_longitude=16,
    minimum_latitude=44.5,
    maximum_latitude=45.5,
)

SPLITS = {
    "train": (date(2021, 1, 1), date(2023, 12, 31)),
    "val":   (date(2024, 1, 1), date(2024, 12, 31)),
    "test":  (date(2025, 1, 1), date(2026, 7, 31)),
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=int, required=True,
                   help="Experiment number, used in the output filename")
    p.add_argument("--features", choices=sorted(EXTRACTORS), required=True,
                   help="nn = point-wise history, nca = 3x3 neighborhood")
    p.add_argument("--seq-length", type=int, default=6)
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    return p.parse_args()

def load_dataset(start: date, end: date, chunk_size=200):
    ds = copernicusmarine.open_dataset(
        dataset_id=DATASET_ID,
        variables=["thetao"],
        start_datetime=start.isoformat(),
        end_datetime=end.isoformat(),
        **BBOX,
    )
    return ds.chunk({"time": chunk_size})


def build_learning_set(ds, extractor, seq_length=6, chunk_size=200, use_gpu=False):
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}", flush=True)

    sst = ds["thetao"]
    if "depth" in sst.dims:
        sst = sst.isel(depth=0)
    sst = sst.astype("float32")

    stats = xr.Dataset({"mean": sst.mean(skipna=True), "std": sst.std(skipna=True)}).compute()
    global_mean = float(stats["mean"])
    global_std = float(stats["std"])
    print(f"Global mean= {global_mean:.4f}", flush=True)
    print(f"Global std= {global_std:.4f}", flush=True)

    X_chunks, Y_chunks = [], []
    total_time = sst.sizes["time"]

    for start in tqdm(range(0, total_time - seq_length, chunk_size),
                      desc="Processing chunks"):
        end = min(total_time, start + chunk_size + seq_length)
        block_np = sst.isel(time=slice(start, end)).compute().values
        block = torch.from_numpy(block_np).float()

        for t in range(block.shape[0] - seq_length):
            #first arg : current block ; second arg: target map
            X_t, Y_t = extractor(block[t : t + seq_length], block[t + seq_length])

            #extractor extracts all inside cells with no discrimination -- nan's arrise.
            nan_in_features = torch.isnan(X_t).any(dim=-1).any(dim=-1)
            nan_in_target = torch.isnan(Y_t).squeeze(-1)
            valid_mask = ~(nan_in_features | nan_in_target)

            X_t = X_t[valid_mask]
            Y_t = Y_t[valid_mask]

            assert X_t.shape[0] == Y_t.shape[0], "X/Y cell count mismatch"
            assert not torch.isnan(X_t).any(), "NaNs remain in X_t after masking"
            assert not torch.isnan(Y_t).any(), "NaNs remain in Y_t after masking"

            X_chunks.append(X_t)
            Y_chunks.append(Y_t)

        del block, block_np

    return torch.cat(X_chunks), torch.cat(Y_chunks), global_mean, global_std


def main():
    args = parse_args()
    extractor = EXTRACTORS[args.features]
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")  # Doc: os.environ.get(key, default)

    for split_name, (sd, ed) in SPLITS.items():
        print(f"\n------ {split_name}: {sd} -> {ed} ------", flush=True)
        ds = load_dataset(sd, ed, chunk_size=args.chunk_size)
        X, Y, global_mean, global_std = build_learning_set(
            ds, 
            extractor,
            seq_length=args.seq_length,
            chunk_size=args.chunk_size,
            use_gpu=args.use_gpu,
        )
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


































