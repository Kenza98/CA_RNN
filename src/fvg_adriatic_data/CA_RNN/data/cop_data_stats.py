import xarray as xr

dataset = xr.open_dataset("test_year.nc")
time = dataset.time
surface_temp = dataset["thetao"]


def print_summary(dataset):
    print(dataset)
    print("Start: ", time.values[0])
    print("End: ", time.values[-1])

    # Time step (we want to check if it is uniform)
    if len(time) > 1:
        step = time.values[1] - time.values[0]
        print("Time step in (sec): ", int(step) * 10**-9)
        print("Time step in (hours) :", int(step) * 10**-9 / 3600)


print_summary(dataset)


def print_coordinates_summary(dataset):
    lat = dataset.latitude
    print(lat.dims)
    lon = dataset.longitude
    print(
        f"Latitude range: {lat.values.min()} to {lat.values.max()} "
        f"({len(lat)} points)"
    )
    print(
        f"Longitude range: {lon.values.min()} to {lon.values.max()} "
        f"({len(lon)} points)"
    )


def grid_is_regular(dataset):
    lat = dataset.latitude
    lon = dataset.longitude
    lat_diff = lat.diff("latitude").values
    lon_diff = lon.diff("longitude").values

    bool_lat = (lat_diff - lat_diff[0] < 10**-45).all()
    bool_lon = (lon_diff - lon_diff[0] < 10**-45).all()

    return bool_lat and bool_lon


def depth_info(dataset):
    depth = dataset.depth
    print(
        f"Depth range: {depth.values.min()} to {depth.values.max()} "
        f"({len(depth)} levels)"
    )
    print("First few depth levels:", depth.values[:5])


def get_temp_info(surface_temp):
    print("thetao shape:", surface_temp.shape)
    print("thetao dims:", surface_temp.dims)


print_summary(dataset)
print_coordinates_summary(dataset)


# Check if grid is regular
if grid_is_regular(dataset):
    print("The grid is regular.")
else:
    print("The grid is not regular.")

depth_info(dataset)
get_temp_info(dataset.thetao)
