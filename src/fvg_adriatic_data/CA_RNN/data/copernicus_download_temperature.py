from copernicusmarine import subset
import netCDF4 as nc

subset(
    dataset_id="med-cmcc-tem-rean-d",
    variables=["thetao"],  # (theta) (o) potential ocean temperature
    minimum_longitude=12,
    maximum_longitude=16,
    minimum_latitude=44.5,
    # maximum_latitude=45.5,  # \\TODO [ask] does this include all the Gulf of Trieste?
    # dates possibles 01/01/1987–31/05/2023
    start_datetime="2020-05-31",
    end_datetime="2022-05-31",
    minimum_depth=1.02,
    maximum_depth=1.02,  # surface temperature only
    output_filename="train_sst.nc",
    # rea = reanalylsis ; all = all years and depths.
)
