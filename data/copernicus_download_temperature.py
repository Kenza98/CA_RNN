from copernicusmarine import subset
import netCDF4 as nc

subset(
    dataset_id="cmems_mod_med_phy-temp_my_4.2km_P1D-m",
    variables=["thetao"],  # (theta) (o) potential ocean temperature
    minimum_longitude=12,
    maximum_longitude=16,
    minimum_latitude=44.5,
    maximum_latitude=45.5,  # \\TODO [ask] does this include all the Gulf of Trieste?
    # dates possibles 01/01/1987–31/01/2026
    start_datetime="2025-01-01",
    end_datetime="2026-01-31",
    minimum_depth=1.02,
    maximum_depth=1.02,  # surface temperature only
    output_filename="test_sst.nc",
    show_progress=True
    # rea = reanalylsis ; all = all years and depths.
)
