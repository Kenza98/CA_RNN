import argparse
import gc
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
OUT_DIR = PROJECT_ROOT / "data"  # data folder at project root
OUT_DIR.mkdir(parents=True, exist_ok=True)  # always play it safe.


def load_dataset(chunk_size=200):
    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_med_phy-temp_my_4.2km_P1D-m",
        variables=["thetao"],
        minimum_longitude=12,
        maximum_longitude=16,
        minimum_latitude=44.5,
        maximum_latitude=45.5,
        start_datetime="2022-01-01",  # "2025-01-01"(test)
        end_datetime="2024-12-31",  # "2026-02-28"(test)
    )
    result = ds.chunk({"time": chunk_size})
    print(type(result))
    return result


def build_learning_set(ds, seq_length=4, chunk_size=200, use_gpu=False):
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}", flush=True)

    sst = ds["thetao"]  # we select the only variable of interest to us, Temperature

    if "depth" in sst.dims:  # we select only sea surface depth
        sst = sst.isel(
            depth=0
        )  # check if 0 here is the index (+1) or the actual depth.

    sst = sst.astype(
        "float32"
    )  # we convert everything to float32 for model compatibility

    print(sst, flush=True)  # prints a summary of the dataset

    global_mean = float(
        sst.mean(skipna=True).compute()
    )  # compute the mean of the variable and skips NA values for accuracy

    print(f"Global mean SST: {global_mean:.4f}", flush=True)

    X_chunks = []
    Y_chunks = []

    total_time = sst.sizes[
        "time"
    ]  # CHECK sst.sizes is a dictionary containing sizes of each dimension
    # CHECK what is tqdm and how the chunking works. why not use a torch dataloader?
    for start in tqdm(
        range(0, total_time - seq_length, chunk_size),
        desc="Processing chunks",
    ):
        end = min(total_time, start + chunk_size + seq_length)
        # get the block chunk in a numpy array
        # then converting to a torch tensor of shape  (T, H, W)
        block_np = sst.isel(time=slice(start, end)).compute().values
        block = torch.from_numpy(block_np).float()

        for t in range(
            block.shape[0] - seq_length
        ):  # create the sequences for each element t in block

            seq_block = block[t : t + seq_length]  # (seq, H, W)
            target_map = block[t + seq_length]  # (H, W)

            # extract 3x3 neighborhoods using torch
            neigh = seq_block.unfold(1, 3, 1).unfold(2, 3, 1)

            # reshape -> (N, seq_length, 9)
            X_t = (
                neigh.contiguous().view(seq_length, -1, 9).permute(1, 0, 2)
            )  # TODO CHECK how .view and .contiguous work in toch

            # targets = center pixels
            Y_t = target_map[1:-1, 1:-1].contiguous().view(-1, 1)

            # handle NaNs
            """
            one nan in neighbors -> kills the sequence step
            one bad sequence step -> kills the whole sample
            """
            nan_in_neighbors = torch.isnan(X_t).any(dim=-1).any(dim=-1)
            nan_in_target = torch.isnan(Y_t).squeeze(-1)
            """
            an invalid x,y if either x or y are nan
            """
            invalid_mask = nan_in_neighbors | nan_in_target
            valid_mask = ~invalid_mask
            X_t = X_t[valid_mask]
            Y_t = Y_t[valid_mask]
            assert not torch.isnan(
                X_t
            ).any(), "NaNs still present in X_t after masking!"
            assert not torch.isnan(
                Y_t
            ).any(), "NaNs still present in Y_t after masking!"

            X_chunks.append(X_t)
            Y_chunks.append(Y_t)

        del block, block_np
        gc.collect()

    X_tensor = torch.cat(X_chunks).to(device)
    Y_tensor = torch.cat(Y_chunks).to(device)

    return X_tensor, Y_tensor


def main():
    sd = date(2022, 1, 1)
    ed = date(2024, 12, 31)
    ds = load_dataset(chunk_size=200)  # just loads the dataset from copernicus

    X_train, Y_train = build_learning_set(
        ds,
        seq_length=4,
        chunk_size=200,
        use_gpu=args.use_gpu,
    )

    print(f"X shape: {X_train.shape}, Y shape: {Y_train.shape}", flush=True)
    output_filepath = OUT_DIR / "sst_train_set.pt"

    torch.save(
        {"X": X_train, "Y": Y_train, "start_time": sd, "end_time": ed},
        output_filepath,
    )

    print(f"Saved: {output_filepath}", flush=True)

    ds.close()


if __name__ == "__main__":
    main()
