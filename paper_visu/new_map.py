import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": ccrs.PlateCarree()})

# basemap features
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAKES, alpha=0.5, edgecolor="blue")
ax.add_feature(cfeature.RIVERS, alpha=0.5, edgecolor="blue")

# study domain rectangle
domain = mpatches.Rectangle(
    xy=(12, 44.5),          # (lon_min, lat_min)
    width=4,                # lon_max - lon_min = 16 - 12
    height=1,               # lat_max - lat_min = 45.5 - 44.5
    transform=ccrs.PlateCarree(),
    linewidth=1.5,
    edgecolor="green",
    facecolor="green",
    alpha=0.3,
    zorder=5
)
ax.add_patch(domain)

# set extent with a bit of padding around the domain
ax.set_extent([9, 20, 42, 48], crs=ccrs.PlateCarree())

# gridlines
gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.7)
gl.xlabel_style = {'size': 14, 'color': 'darkred'}
gl.ylabel_style = {'size': 14, 'color': 'darkred'}
gl.top_labels = False
gl.right_labels = False

# cities
cities = {
    "Trieste": (13.77, 45.65),
    "Venice":  (12.32, 45.44),
    "Rijeka":  (14.44, 45.33),
    "Ancona":  (13.51, 43.62),
}
for city, (lon, lat) in cities.items():
    ax.plot(lon, lat, marker="o", color="black", markersize=3.5,
            transform=ccrs.PlateCarree())
    ax.text(lon + 0.1, lat + 0.1, city, fontsize=12,
            transform=ccrs.PlateCarree(), color="black")

ax.set_title("Figure 1: Illustration of the study domain.", fontsize=16, color="black")

plt.tight_layout()
plt.show()