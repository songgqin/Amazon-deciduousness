"""Reproduce the manuscript-facing main-figure inputs and visual checks.

This entry point is deliberately small and deterministic. It provides a
repository-relative, non-interactive check for the raster inputs and renders
the Figure 2 map/time-series composite plus the Figure 3 three-driver map from
the released inputs.

Run from any working directory with the repository Python environment::

    python code/reproduce_main_figures.py

The generated files are written below ``outputs/figures`` and are ignored by
Git.
"""

from __future__ import annotations

import argparse
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

# Values read from the Word manuscript's Nature-revision Figure 3 panel e.
# The order follows the ternary raster bands: Light, Hydroclimate, Soil.
FIGURE3_TARGET_PROPORTIONS = np.array([0.152, 0.516, 0.332], dtype=np.float32)


def read_raster(path: Path, dtype=np.float32) -> tuple[np.ndarray, tuple[float, ...]]:
    """Read a GDAL raster with bands last and convert common nodata values."""

    dataset = gdal.Open(str(path))
    if dataset is None:
        raise FileNotFoundError(path)
    array = dataset.ReadAsArray()
    if array.ndim == 3:
        array = np.moveaxis(array, 0, -1)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    elif not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
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


def figure2_site_data(lag_path: Path | None = None) -> pd.DataFrame:
    """Match the local revision: native 3x3 patches and SD across three years.

    Means use the supplied three-year mean rasters. Error bars use the SD
    (ddof=0) of each year's spatial mean, not the spatial SD of the mean raster.
    EVI and deciduousness are located on their respective native grids.
    """
    years = (2019, 2020, 2021)
    locations = pd.read_csv(DATA / "phenocam" / "ATTO_RJA_Location.csv").set_index("Site")
    dec, dec_geo = read_raster(DATA / "deciduousness" / "Figure2_Deciduousness_Seasonality.tif")
    dec[dec > 1000] = np.nan
    dec /= 1000.0
    evi, evi_geo = read_raster(DATA / "seasonality" / "BRDF_EVI_3years_mean.tif", dtype=None)
    rain, _ = read_raster(DATA / "drivers" / "inputs" / "hydroclimate_precipitation_ERA.tif")
    rain = cv2.resize(rain, (786, 650)) * 1000.0
    lag, _ = read_raster(lag_path or DATA / "seasonality" / "time_lag_map.tif")
    annual = {"deciduousness": [], "evi": []}
    for year in years:
        d, geo = read_raster(DATA / "deciduousness" / f"Composite_Data_{year}_5km_gf.tif")
        if d.shape != dec.shape or not np.allclose(geo, dec_geo, rtol=0, atol=1e-8):
            raise ValueError(f"Deciduousness grid differs in {year}")
        d[d > 1000] = np.nan
        annual["deciduousness"].append(d / 1000.0)
        e, geo = read_raster(DATA / "seasonality" / f"BRDF_EVI_{year}.tif", dtype=None)
        if e.shape != evi.shape or not np.allclose(geo, evi_geo, rtol=0, atol=1e-8):
            raise ValueError(f"EVI grid differs in {year}")
        annual["evi"].append(e)

    def patch(array, geo, lat, lon):
        row, col = int((lat - geo[3]) / geo[5]), int((lon - geo[0]) / geo[1])
        if row < 1 or col < 1 or row + 1 >= array.shape[0] or col + 1 >= array.shape[1]:
            raise ValueError("Site does not have a complete 3x3 raster neighborhood")
        return array[row - 1:row + 2, col - 1:col + 2]

    annual = {name: np.stack(values, axis=0) for name, values in annual.items()}
    records = []
    for site in ("ATTO", "RJA"):
        lat, lon = locations.loc[site, ["Lat", "Lon"]].astype(float)
        d = np.nanmean(patch(dec, dec_geo, lat, lon), axis=(0, 1))
        e = np.nanmean(patch(evi, evi_geo, lat, lon), axis=(0, 1))
        drow, dcol = int((lat - dec_geo[3]) / dec_geo[5]), int((lon - dec_geo[0]) / dec_geo[1])
        erow, ecol = int((lat - evi_geo[3]) / evi_geo[5]), int((lon - evi_geo[0]) / evi_geo[1])
        dy = np.nanmean(annual["deciduousness"][:, drow - 1:drow + 2, dcol - 1:dcol + 2], axis=(1, 2))
        ey = np.nanmean(annual["evi"][:, erow - 1:erow + 2, ecol - 1:ecol + 2], axis=(1, 2))
        p = np.nanmean(patch(rain, dec_geo, lat, lon), axis=(0, 1))
        l = patch(lag, dec_geo, lat, lon)
        ds, es = np.nanstd(dy, axis=0, ddof=0), np.nanstd(ey, axis=0, ddof=0)
        if not all(np.isfinite(a).all() for a in (d, e, dy, ey, p)):
            raise ValueError(f"Incomplete monthly support at {site}")
        for month in range(12):
            record = dict(site=site, month=month + 1, deciduousness_mean=float(d[month]),
                          deciduousness_interannual_sd=float(ds[month]), evi_mean=float(e[month]),
                          evi_interannual_sd=float(es[month]), precipitation_mm=float(p[month]),
                          dry_month=bool(p[month] < 100), lag_mean_month=float(np.nanmean(l)),
                          lag_spatial_sd_month=float(np.nanstd(l)), n_years=3, sd_ddof=0)
            for index, year in enumerate(years):
                record[f"deciduousness_{year}"] = float(dy[index, month])
                record[f"evi_{year}"] = float(ey[index, month])
            records.append(record)
    return pd.DataFrame.from_records(records)


def figure2(output_dir: Path | None = None, lag_path: Path | None = None) -> Path:
    """Render the Figure 2 map and ATTO/RJA seasonal-cycle composite."""

    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    dec, dec_geo = read_raster(RELEASE / "Deciduousness_Seasonality_Amazon.tif")
    lag, _ = read_raster(lag_path or DATA / "seasonality" / "time_lag_map.tif")
    mask, _ = read_raster(RELEASE / "MCD12Q1_Amazon.tif")

    # The public release contains the native 785-column products.  The
    # manuscript map grid is 786 columns, as in the original materials code.
    width, height = 786, 650
    dec = resize_stack(dec, width, height) / 1000.0
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    dec[mask != 2] = np.nan

    valid = np.isfinite(dec).any(axis=2)
    amplitude = np.full(dec.shape[:2], np.nan, dtype=np.float32)
    amplitude[valid] = np.nanmax(dec[valid], axis=1) - np.nanmin(dec[valid], axis=1)
    extent = [dec_geo[0], dec_geo[0] + dec_geo[1] * width, dec_geo[3] + dec_geo[5] * height, dec_geo[3]]

    output_dir = FIGURES if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sites = figure2_site_data(lag_path)
    sites.to_csv(output_dir / "Main_Figure2_site_monthly.csv", index=False)
    locations = pd.read_csv(DATA / "phenocam" / "ATTO_RJA_Location.csv").set_index("Site").loc[["ATTO", "RJA"]]
    site_names = ["ATTO", "RJA"]
    month_labels = list("JFMAMJJASOND")
    fig = plt.figure(figsize=(11.5, 7.0), dpi=220)
    projection = ccrs.PlateCarree()

    map_axes = [fig.add_subplot(2, 2, 1, projection=projection), fig.add_subplot(2, 2, 2, projection=projection)]
    for ax, data, title, cmap, norm in [
        (map_axes[0], amplitude, "A  Deciduousness amplitude", "YlOrRd", mcolors.Normalize(0, 1)),
        (map_axes[1], lag, "B  Asynchrony", "RdBu_r", mcolors.PowerNorm(gamma=1.1, vmin=0, vmax=6)),
    ]:
        ax.add_feature(__import__("cartopy.feature", fromlist=["LAND"]).LAND, facecolor="white", zorder=0)
        artist = ax.imshow(data, origin="upper", extent=extent, transform=projection, cmap=cmap, norm=norm)
        for name, location in locations.iterrows():
            ax.plot(location.Lon, location.Lat, marker="*", color="black", markersize=7, transform=projection)
            ax.text(location.Lon - 0.5, location.Lat - 1.8, name, fontsize=7, transform=projection)
        colorbar = fig.colorbar(artist, ax=ax, orientation="horizontal", pad=0.07, fraction=0.04)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label("Deciduousness amplitude" if ax is map_axes[0] else "Asynchrony (month)", fontsize=8)
        inset = ax.inset_axes([0.69, 0.08, 0.26, 0.22])
        values = data[np.isfinite(data)]
        bins = np.linspace(0, 1, 21) if ax is map_axes[0] else np.arange(-0.5, 7.5, 1)
        inset.hist(values, bins=bins, weights=np.ones(len(values)) / len(values), color="0.55", edgecolor="white", linewidth=0.3)
        inset.set_ylabel("Proportion", fontsize=5)
        inset.tick_params(labelsize=5, length=2)
        inset.spines[["top", "right"]].set_visible(False)
        ax.set_extent([-80, -44, -22, 10], crs=projection)
        ax.set_xticks(np.arange(-80, -40, 10), crs=projection)
        ax.set_yticks(np.arange(-20, 11, 10), crs=projection)
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.set_title(title, loc="left", fontsize=10)
        ax.tick_params(labelsize=7)

    season_axes = [fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)]
    for panel_index, (ax, site) in enumerate(zip(season_axes, site_names)):
        values = sites.loc[sites.site == site].sort_values("month")
        d, e, p = (values[name].to_numpy() for name in ("deciduousness_mean", "evi_mean", "precipitation_mm"))
        x = np.arange(12)
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 30))
        ax.errorbar(x, d, yerr=values.deciduousness_interannual_sd, color="black", marker="o", lw=1, ms=3)
        ax2.errorbar(x, e, yerr=values.evi_interannual_sd, color="green", marker="o", lw=1, ms=3)
        ax3.bar(x, p, color="#0C7BDC", alpha=0.35, width=0.85)
        for month in np.flatnonzero(p < 100):
            ax.axvspan(month - 0.5, month + 0.5, color="#D8D3C5", alpha=0.3, zorder=0)
        ax.set_title(f"{chr(67 + panel_index)}  {site_names[panel_index]}", loc="left", fontsize=10)
        ax.set_ylabel("Deciduousness", fontsize=8)
        ax2.set_ylabel("EVI", color="green", fontsize=8)
        ax3.set_ylabel("Precipitation (mm)", color="#0C7BDC", fontsize=8)
        ax.set_xticks(x, month_labels, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax2.tick_params(axis="y", labelsize=7, colors="green")
        ax3.tick_params(axis="y", labelsize=7, colors="#0C7BDC")
        ax.text(0.03, 0.92, f"Δt = {values.lag_mean_month.iloc[0]:.1f} ± {values.lag_spatial_sd_month.iloc[0]:.1f} month\nMAP = {np.nansum(p):.0f} mm/year", transform=ax.transAxes, fontsize=7, va="top")

    fig.tight_layout()
    out = output_dir / "Main_Figure2_reproduced.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure3_map(strict: bool = False, output_dir: Path | None = None, driver_path: Path | None = None) -> Path:
    """Render the released three-driver SHAP composition used by Figure 3e."""

    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

    output_dir = FIGURES if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    driver_path = driver_path or DATA / "drivers" / "asynchrony_driver_map_3type.tif"
    drivers, geo = read_raster(driver_path)
    rgb = np.array([[208, 28, 139], [65, 182, 196], [253, 184, 99]], dtype=np.float32) / 255.0
    image = np.nansum(drivers[..., None] * rgb[None, None, :, :], axis=2)
    image[np.all(np.isnan(drivers), axis=2)] = 1.0
    valid = np.isfinite(drivers).all(axis=2) & (np.nansum(drivers, axis=2) > 0)
    proportions = np.nanmean(drivers[valid], axis=0)
    metrics = {
        "source_raster": str(driver_path.relative_to(ROOT) if driver_path.is_relative_to(ROOT) else driver_path),
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
    out = output_dir / "Main_Figure3e_reproduced.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    with (output_dir / "Main_Figure3e_reproduced_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    print("Figure 3e driver proportions:", dict(zip(names.tolist(), np.round(proportions, 4).tolist())))
    if strict and not metrics["rounded_percent_match"]:
        raise RuntimeError(
            "Figure 3e QA failed: the repository raster gives "
            f"{np.round(proportions * 100, 1).tolist()}% (Light, Hydroclimate, Soil), "
            "but the Word Figure 3 panel e gives "
            f"{np.round(FIGURE3_TARGET_PROPORTIONS * 100, 1).tolist()}%. "
            "The exact revision raster is missing or differs from the checked-in driver raster."
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    print("Repository root:", ROOT)
    outputs = [figure2(args.output_dir), figure3_map(strict=True, output_dir=args.output_dir)]
    for path in outputs:
        print("Wrote:", path)


if __name__ == "__main__":
    main()
