import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
OUT_DIR = PROJECT_ROOT / "paper_visu"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})

ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)

ax.set_extent([12, 16, 43.5, 46.5], crs=ccrs.PlateCarree())

gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.7)
gl.xlabel_style = {'size': 12, 'color': 'black'}
gl.ylabel_style = {'size': 12, 'color': 'black'}
gl.top_labels = False
gl.right_labels = False

lon, lat = 13.77, 45.65
ax.plot(lon, lat, marker="o", color="black", markersize=5, transform=ccrs.PlateCarree())
ax.text(lon + 0.1, lat + 0.1, "Trieste", fontsize=13, transform=ccrs.PlateCarree())

#ax.set_title("Figure 1: Illustration of the study domain.", fontsize=14)

fp = OUT_DIR / "map.png"
fig.savefig(fp, dpi=300, bbox_inches="tight")
