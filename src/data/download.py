"""Download raw Copernicus SST for each split, once, to local netCDF.

Idempotent by default: skips files that already exist. Pass --force to
re-download and overwrite.

Usage:
    python -m src.data.download
    python -m src.data.download --force
"""

import argparse
from datetime import date
from pathlib import Path

import copernicusmarine
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


DATASET_ID = "cmems_mod_med_phy-temp_my_4.2km_P1D-m"  # reanalysis daily product
BBOX = dict(  # adriatic sea bounding box
    minimum_longitude=12,
    maximum_longitude=16,
    minimum_latitude=44.5,
    maximum_latitude=45.5,
)

SPLITS = {  # experiment dates for train / val / test
    "train": (date(2021, 1, 1), date(2023, 12, 31)),
    "val": (date(2024, 1, 1), date(2024, 12, 31)),
    "test": (date(2025, 1, 1), date(2026, 7, 31)),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    return p.parse_args()


def raw_path(split_name: str) -> Path:
    return RAW_DIR / f"sst_{split_name}.nc"


def download_split(
    split_name: str, start: date, end: date, force: bool = False
) -> Path:
    out = raw_path(split_name)
    if out.exists() and not force:
        print(f"[skip] {out.name} already present", flush=True)
        return out

    print(f"[fetch] {split_name}: {start} -> {end}", flush=True)
    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        variables=["thetao"],
        start_datetime=start.isoformat(),
        end_datetime=end.isoformat(),
        minimum_longitude=BBOX["minimum_longitude"],
        maximum_longitude=BBOX["maximum_longitude"],
        minimum_latitude=BBOX["minimum_latitude"],
        maximum_latitude=BBOX["maximum_latitude"],
        minimum_depth=0,
        maximum_depth=2,
        output_directory=str(RAW_DIR),
        output_filename=out.name,
        overwrite=True,
    )
    print(f"[done] {out.name} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out


def main():
    args = parse_args()
    for split_name, (sd, ed) in SPLITS.items():
        download_split(split_name, sd, ed, force=args.force)


if __name__ == "__main__":
    main()
