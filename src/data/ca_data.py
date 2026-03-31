import argparse
from pathlib import Path

import copernicusmarine
import torch
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument("--use-gpu", action="store_true")
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_sst(start="2025-01-01", end="2026-02-28"):
    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_med_phy-temp_my_4.2km_P1D-m",
        variables=["thetao"],
        minimum_longitude=12,
        maximum_longitude=16,
        minimum_latitude=43,
        maximum_latitude=46,
        start_datetime=start,
        end_datetime=end,
    )  # this dataset contains many variables.
    print("Dataset type:", type(ds))
    sst = ds["thetao"]  # we only need temperature
    print("SST type : ", type(sst))
    if "depth" in sst.dims:  # true always, but safety measure
        sst = sst.isel(depth=0)  # only the first layer for studying SST
    print(
        f"The current type of sst dataset is :{type(sst)}\n  \
         and the type of the data is:  {sst.dtype}\n"
    )

    print(f"SST shape: {sst.sizes}")  # e.g. {'time': T, 'latitude': H, 'longitude': W}

    return sst


# helper function to print NaN percentage
def nan_ratio(tensor, name="tensor"):
    total = tensor.numel()
    nans = torch.isnan(tensor).sum().item()
    print(f"{name}: {nans}/{total} NaN ({100 * nans / total:.1f}%)")


def main():
    sst = load_sst()
    # don't i want tensor \down?
    sst_np = sst.compute().values  # shape (T, H, W)
    sst_tensor = torch.from_numpy(sst_np)
    lats = torch.from_numpy(sst.latitude.values)  # (24,)
    lons = torch.from_numpy(sst.longitude.values)  # (97,)
    # nan_ratio(lats, "latitudes")
    # nan_ratio(lons, "longitudes")
    nan_ratio(sst_tensor, "full data")

    # tighten the box before saving data
    first_grid = sst_tensor[0]
    water_mask = ~torch.isnan(first_grid)

    water_lat_indices = water_mask.any(dim=1).nonzero().squeeze()
    water_lon_indices = water_mask.any(dim=0).nonzero().squeeze()

    # latitude indices
    l_min = water_lat_indices[0]
    l_max = water_lat_indices[-1]
    # longitude indices
    L_min = water_lon_indices[0]
    L_max = water_lon_indices[-1]

    lats_cropped = lats[l_min : l_max + 1]
    lons_cropped = lons[L_min : L_max + 1]

    sst_tensor_cropped = sst_tensor[:, l_min : l_max + 1, L_min : L_max + 1]
    nan_ratio(sst_tensor_cropped, "full data")

    # print(sst_tensor_cropped.shape)
    # print(lats_cropped[0], lats_cropped[-1])
    # print(lons_cropped[0], lons_cropped[-1])

    fp = DATA_DIR / "ca_data.pt"
    # print(f"Type of dataset now: {type(sst_tensor_cropped)}\n")

    torch.save(
        {
            "full_data": sst_tensor_cropped,
            "latitudes": lats_cropped,
            "longitudes": lons_cropped,
        },
        fp,
    )


if __name__ == "__main__":
    main()
