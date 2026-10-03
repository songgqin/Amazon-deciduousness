"""Reproduce the manuscript-facing main-figure inputs and visual checks.

This entry point is deliberately small and deterministic. It provides a
repository-relative, non-interactive check for the raster inputs and renders
the Figure 2 map/time-series composite plus the Figure 3 three-driver map from
the released inputs.

Run from any working directory with the repository Python environment::

    python code/reproduce_main_figures.py

The generated files are written below ``outputs/figures`` and are ignored by
Git.  The source reference figures are kept separately in ``reference_figures``
for visual comparison.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from osgeo import gdal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RELEASE = DATA / "maintext_release"
FIGURES = ROOT / "outputs" / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

# Values read from the Word manuscript's Nature-revision Figure 3 panel e.
# The order follows the ternary raster bands: Light, Hydroclimate, Soil.
FIGURE3_TARGET_PROPORTIONS = np.array([0.152, 0.516, 0.332], dtype=np.float32)


def read_raster(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    """Read a GDAL raster with bands last and convert common nodata values."""

    dataset = gdal.Open(str(path))
    if dataset is None:
        raise FileNotFoundError(path)
    array = dataset.ReadAsArray()
    if array.ndim == 3:
        array = np.moveaxis(array, 0, -1)
    array = array.astype(np.float32, copy=False)
    nodata = dataset.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        array = array.copy()
        array[array == nodata] = np.nan
    array[array == 65535] = np.nan
    array[array == -9999] = np.nan
    return array, dataset.GetGeoTransform()


def resize_stack(array: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a monthly or multiband stack while preserving band order."""

    if array.ndim == 2:
        return cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.stack(
        [cv2.resize(array[:, :, band], (width, height), interpolation=cv2.INTER_LINEAR)
         for band in range(array.shape[2])],
        axis=2,
    )


def add_boundary(ax) -> None:
    """Draw the released Amazon three-region boundary when Cartopy is present."""

    try:
        import cartopy.crs as ccrs
        import cartopy.io.shapereader as shpreader

        reader = shpreader.Reader(str(DATA / "boundaries" / "Amazon_ThreeRegions_Clip.shp"))
        ax.add_geometries(reader.geometries(), ccrs.PlateCarree(), facecolor="none", edgecolor="black", linewidth=0.45)
        reader.close()
    except ImportError:
        return


def figure2() -> Path:
    """Render the Figure 2 map and ATTO/RJA seasonal-cycle composite."""

    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    dec, dec_geo = read_raster(RELEASE / "Deciduousness_Seasonality_Amazon.tif")
    evi, _ = read_raster(RELEASE / "BRDF_EVI.tif")
    rainfall, _ = read_raster(RELEASE / "Environmental_Variables" / "hydroclimate_precipitation_ERA.tif")
    lag, _ = read_raster(DATA / "seasonality" / "time_lag_map_0521.tif")
    mask, _ = read_raster(RELEASE / "MCD12Q1_Amazon.tif")

    # The public release contains the native 785-column products.  The
    # manuscript map grid is 786 columns, as in the original materials code.
    width, height = 786, 650
    dec = resize_stack(dec, width, height) / 1000.0
    evi = resize_stack(evi, width, height)
    rainfall = resize_stack(rainfall, width, height) * 1000.0
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    evi[(evi < 0) | (evi > 1)] = np.nan
    dec[mask != 2] = np.nan
    evi[mask != 2] = np.nan
    rainfall[mask != 2] = np.nan

    amplitude = np.nanmax(dec, axis=2) - np.nanmin(dec, axis=2)
    extent = [dec_geo[0], dec_geo[0] + dec_geo[1] * width, dec_geo[3] + dec_geo[5] * height, dec_geo[3]]

    locations = pd.read_csv(DATA / "phenocam" / "ATTO_RJA_Location.csv")
    site_names = ["ATTO", "RJA"]
    month_labels = list("JFMAMJJASOND")
    fig = plt.figure(figsize=(11.5, 7.0), dpi=220)
    projection = ccrs.PlateCarree()

    map_axes = [fig.add_subplot(2, 2, 1, projection=projection), fig.add_subplot(2, 2, 2, projection=projection)]
    for ax, data, title, cmap, norm in [
        (map_axes[0], amplitude, "A  Deciduousness amplitude", "YlOrRd", mcolors.Normalize(0, 1)),
        (map_axes[1], lag, "B  Asynchrony", "RdBu_r", mcolors.BoundaryNorm(np.arange(8), 7)),
    ]:
        ax.add_feature(__import__("cartopy.feature", fromlist=["LAND"]).LAND, facecolor="white", zorder=0)
        ax.imshow(data, origin="upper", extent=extent, transform=projection, cmap=cmap, norm=norm)
        add_boundary(ax)
        ax.set_extent([-80, -44, -22, 10], crs=projection)
        ax.set_xticks(np.arange(-80, -40, 10), crs=projection)
        ax.set_yticks(np.arange(-20, 11, 10), crs=projection)
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_title(title, loc="left", fontsize=10)
        ax.tick_params(labelsize=7)

    season_axes = [fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)]
    for panel_index, (ax, row) in enumerate(zip(season_axes, locations.itertuples(index=False))):
        lat, lon = float(row.Lat), float(row.Lon)
        col = int((lon - dec_geo[0]) / dec_geo[1])
        r = int((lat - dec_geo[3]) / dec_geo[5])
        r0, r1 = max(1, r - 1), min(height - 1, r + 2)
        c0, c1 = max(1, col - 1), min(width - 1, col + 2)
        d = np.nanmean(dec[r0:r1, c0:c1, :], axis=(0, 1))
        e = np.nanmean(evi[r0:r1, c0:c1, :], axis=(0, 1))
        p = np.nanmean(rainfall[r0:r1, c0:c1, :], axis=(0, 1))
        l = lag[r0:r1, c0:c1]
        x = np.arange(12)
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 30))
        ax.errorbar(x, d, yerr=np.nanstd(dec[r0:r1, c0:c1, :], axis=(0, 1)), color="black", marker="o", lw=1, ms=3)
        ax2.plot(x, e, color="green", marker="o", lw=1, ms=3)
        ax3.bar(x, p, color="#0C7BDC", alpha=0.35, width=0.85)
        ax.set_title(f"{chr(67 + panel_index)}  {site_names[panel_index]}", loc="left", fontsize=10)
        ax.set_ylabel("Deciduousness", fontsize=8)
        ax2.set_ylabel("EVI", color="green", fontsize=8)
        ax3.set_ylabel("Precipitation (mm)", color="#0C7BDC", fontsize=8)
        ax.set_xticks(x, month_labels, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax2.tick_params(axis="y", labelsize=7, colors="green")
        ax3.tick_params(axis="y", labelsize=7, colors="#0C7BDC")
        ax.text(0.03, 0.92, f"Δt = {np.nanmean(l):.1f} ± {np.nanstd(l):.1f} month\nMAP = {np.nansum(p):.0f} mm yr$^{{-1}}", transform=ax.transAxes, fontsize=7, va="top")

    fig.tight_layout()
    out = FIGURES / "Main_Figure2_reproduced.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure3_map(strict: bool = False) -> Path:
    """Render the released three-driver SHAP composition used by Figure 3e."""

    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    drivers, geo = read_raster(DATA / "drivers" / "Asynchrony_shap_map_3type_drivers_0818.tif")
    rgb = np.array([[208, 28, 139], [65, 182, 196], [253, 184, 99]], dtype=np.float32) / 255.0
    image = np.nansum(drivers[..., None] * rgb[None, None, :, :], axis=2)
    image[np.all(np.isnan(drivers), axis=2)] = 1.0
    valid = np.isfinite(drivers).all(axis=2) & (np.nansum(drivers, axis=2) > 0)
    proportions = np.nanmean(drivers[valid], axis=0)
    metrics = {
        "source_raster": str((DATA / "drivers" / "Asynchrony_shap_map_3type_drivers_0818.tif").relative_to(ROOT)),
        "band_order": ["Light", "Hydroclimate", "Soil"],
        "valid_pixels": int(valid.sum()),
        "proportions": {name: float(value) for name, value in zip(["Light", "Hydroclimate", "Soil"], proportions)},
        "manuscript_target_percent": {name: float(value * 100) for name, value in zip(["Light", "Hydroclimate", "Soil"], FIGURE3_TARGET_PROPORTIONS)},
        "rounded_percent_match": bool(np.array_equal(np.round(proportions * 100, 1), np.round(FIGURE3_TARGET_PROPORTIONS * 100, 1))),
    }

    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(7.2, 5.2), dpi=220)
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.imshow(image, origin="upper", extent=[geo[0], geo[0] + geo[1] * image.shape[1], geo[3] + geo[5] * image.shape[0], geo[3]], transform=projection)
    add_boundary(ax)
    ax.set_extent([-80, -44, -22, 10], crs=projection)
    ax.set_xticks(np.arange(-80, -40, 10), crs=projection)
    ax.set_yticks(np.arange(-20, 11, 10), crs=projection)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=7)
    inset = ax.inset_axes([0.62, 0.08, 0.32, 0.29])
    order = np.argsort(proportions)[::-1]
    names = np.array(["Light", "Hydroclimate", "Soil"])
    x = np.arange(len(order))
    inset.bar(x, proportions[order], color=rgb[order], edgecolor="black")
    inset.set_ylim(0, 0.7)
    inset.set_xticks(x, names[order])
    inset.tick_params(axis="x", labelrotation=35, labelsize=6)
    inset.tick_params(axis="y", labelsize=6)
    inset.set_ylabel("Proportion", fontsize=6)
    for idx, value in enumerate(proportions[order]):
        inset.text(idx, value + 0.015, f"{value * 100:.1f}%", ha="center", fontsize=6)
    fig.tight_layout()
    out = FIGURES / "Main_Figure3e_reproduced.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    with (FIGURES / "Main_Figure3e_reproduced_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    print("Figure 3e driver proportions:", dict(zip(names.tolist(), np.round(proportions, 4).tolist())))
    if strict and not metrics["rounded_percent_match"]:
        raise RuntimeError(
            "Figure 3e QA failed: the repository raster gives "
            f"{np.round(proportions * 100, 1).tolist()}% (Light, Hydroclimate, Soil), "
            "but the Word Figure 3 panel e gives "
            f"{np.round(FIGURE3_TARGET_PROPORTIONS * 100, 1).tolist()}%. "
            "The exact revision raster is missing or differs from the checked-in 0818 raster."
        )
    return out


def main() -> None:
    print("Repository root:", ROOT)
    outputs = [figure2(), figure3_map(strict=True)]
    for path in outputs:
        print("Wrote:", path)


if __name__ == "__main__":
    main()
