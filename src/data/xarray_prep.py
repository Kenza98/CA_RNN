import argparse
import gc
import os
from pathlib import Path
import copernicusmarine
import torch
import xarray as xr
from tqdm import tqdm
from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-gpu",
    action="store_true",
    help="Use GPU if available",
)
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

SPLITS = {
    "train": (date(2015, 1, 1), date(2020, 12, 31)),
    "val":   (date(2021, 1, 1), date(2022, 12, 31)),
    "test":  (date(2023, 1, 1), date(2026, 2, 28)),
}

def load_dataset(start: date, end: date, chunk_size=200):
    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_med_phy-temp_my_4.2km_P1D-m",
        variables=["thetao"],
        minimum_longitude=12,
        maximum_longitude=16,
        minimum_latitude=44.5,
        maximum_latitude=45.5,
        start_datetime=start.isoformat(),
        end_datetime=end.isoformat(),
    )
    return ds.chunk({"time": chunk_size})


def build_learning_set(ds, seq_length=4, chunk_size=200, use_gpu=False):
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}", flush=True)

    sst = ds["thetao"]
    if "depth" in sst.dims:
        sst = sst.isel(depth=0)
    sst = sst.astype("float32")
    print(sst, flush=True)

    global_mean = float(sst.mean(skipna=True).compute())
    print(f"Global mean SST: {global_mean:.4f}", flush=True)

    X_chunks = []
    Y_chunks = []
    total_time = sst.sizes["time"]

    for start in tqdm(
        range(0, total_time - seq_length, chunk_size),
        desc="Processing chunks",
    ):
        end = min(total_time, start + chunk_size + seq_length)
        block_np = sst.isel(time=slice(start, end)).compute().values
        block = torch.from_numpy(block_np).float()

        for t in range(block.shape[0] - seq_length):
            seq_block = block[t : t + seq_length]
            target_map = block[t + seq_length]

            neigh = seq_block.unfold(1, 3, 1).unfold(2, 3, 1)
            X_t = neigh.contiguous().view(seq_length, -1, 9).permute(1, 0, 2)
            Y_t = target_map[1:-1, 1:-1].contiguous().view(-1, 1)

            nan_in_neighbors = torch.isnan(X_t).any(dim=-1).any(dim=-1)
            nan_in_target = torch.isnan(Y_t).squeeze(-1)
            invalid_mask = nan_in_neighbors | nan_in_target
            valid_mask = ~invalid_mask

            X_t = X_t[valid_mask]
            Y_t = Y_t[valid_mask]

            assert not torch.isnan(X_t).any(), "NaNs still present in X_t after masking!"
            assert not torch.isnan(Y_t).any(), "NaNs still present in Y_t after masking!"

            X_chunks.append(X_t)
            Y_chunks.append(Y_t)

        del block, block_np
        gc.collect()

    X_tensor = torch.cat(X_chunks).to(device)
    Y_tensor = torch.cat(Y_chunks).to(device)
    return X_tensor, Y_tensor


def main():
    for split_name, (sd, ed) in SPLITS.items():
        print(f"\n=== Generating {split_name} set: {sd} -> {ed} ===", flush=True)
        ds = load_dataset(sd, ed, chunk_size=200)
        X, Y = build_learning_set(ds, seq_length=6, chunk_size=200, use_gpu=args.use_gpu)
        print(f"X shape: {X.shape}, Y shape: {Y.shape}", flush=True)

        filename = f"sst_{split_name}_set_fixed_{SLURM_JOB_ID}.pt"
        output_filepath = OUT_DIR / filename
        torch.save(
            {"X": X, "Y": Y, "start_time": sd, "end_time": ed},
            output_filepath,
        )
        print(f"Saved: {output_filepath}", flush=True)
        ds.close()
        del X, Y
        gc.collect()


if __name__ == "__main__":
    main()
