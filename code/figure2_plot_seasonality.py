"""Figure 2 seasonality, deciduousness amplitude and categorical time-lag maps.

All default inputs come from this repository. Error bars default to the
manuscript's interannual SD; --error-bars spatial reproduces the supplied
legacy plot's spatial SD instead. No correlation-map input is required.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from reproduce_main_figures import read_raster, add_boundary, figure2_site_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def patch(array, geo, lat, lon):
    row = int((lat - geo[3]) / geo[5])
    col = int((lon - geo[0]) / geo[1])
    if not (1 <= row < array.shape[0] - 1 and 1 <= col < array.shape[1] - 1):
        raise ValueError("Site does not have a complete 3x3 neighborhood")
    return array[row - 1:row + 2, col - 1:col + 2]


def spatial_site_data(locations, forest, lag):
    # Preserve the legacy EVI resize and its original indexing transform.
    dec, geo = read_raster(DATA / "deciduousness/Figure2_Deciduousness_Seasonality.tif")
    dec[dec > 1000] = np.nan
    dec[~forest] = np.nan
    evi, egeo = read_raster(DATA / "seasonality/BRDF_EVI_3years_mean.tif", dtype=None)
    evi = cv2.resize(evi, (786, 650))
    rain, _ = read_raster(DATA / "drivers/inputs/hydroclimate_precipitation_ERA.tif")
    rain = cv2.resize(rain, (786, 650)) * 1000.0
    records = []
    for site, loc in locations.iterrows():
        d = patch(dec, geo, loc.Lat, loc.Lon)
        e = patch(evi, egeo, loc.Lat, loc.Lon)
        p = patch(rain, geo, loc.Lat, loc.Lon)
        l = patch(lag, geo, loc.Lat, loc.Lon)
        for month in range(12):
            records.append(dict(
                site=site, month=month + 1,
                deciduousness_mean=float(np.nanmean(d, axis=(0, 1))[month] / 1000.0),
                deciduousness_sd=float(np.nanstd(d, axis=(0, 1), ddof=0)[month] / 1000.0),
                evi_mean=float(np.nanmean(e, axis=(0, 1))[month]),
                evi_sd=float(np.nanstd(e, axis=(0, 1), ddof=0)[month]),
                precipitation_mm=float(np.nanmean(p, axis=(0, 1))[month]),
                lag_mean_month=float(np.nanmean(l)),
                lag_spatial_sd_month=float(np.nanstd(l))))
    return pd.DataFrame(records)


def save_figure(fig, output, stem):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def seasonality(sites, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.8))
    fig.subplots_adjust(left=0.065, right=0.89, bottom=0.16, top=0.86, wspace=0.65)
    x = np.arange(12)
    for ax, site in zip(axes, ("ATTO", "RJA")):
        v = sites.loc[sites.site == site].sort_values("month")
        evi_ax, rain_ax = ax.twinx(), ax.twinx()
        rain_ax.spines["right"].set_position(("outward", 44))
        rain_ax.bar(x, v.precipitation_mm, width=0.9, color="#0C7BDC", alpha=0.45)
        for month in np.flatnonzero(v.precipitation_mm.to_numpy() < 100):
            rain_ax.axvspan(month - 0.45, month + 0.45, color="grey", alpha=0.21, lw=0)
        rain_ax.set_ylim(0, 1000)
        rain_ax.set_yticks([0, 100, 200, 300, 400])
        rain_ax.spines["right"].set_bounds(0, 400)
        rain_ax.set_ylabel("Precipitation (mm)", color="#0C7BDC")
        rain_ax.yaxis.set_label_coords(1.30, 0.20)
        rain_ax.tick_params(axis="y", colors="#0C7BDC")
        rain_ax.spines["right"].set_color("#0C7BDC")
        ax.set_zorder(2)
        evi_ax.set_zorder(3)
        ax.patch.set_visible(False)
        evi_ax.patch.set_visible(False)
        ax.errorbar(x, v.deciduousness_mean * 100, yerr=v.deciduousness_sd * 100,
                    fmt="o-", color="black", ms=4, lw=1, capsize=2)
        evi_ax.errorbar(x, v.evi_mean, yerr=v.evi_sd, fmt="o-", color="green",
                       ms=4, lw=1, capsize=2)
        ax.set_ylim(-6, 40)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_ylabel("Deciduousness (%)")
        ax.set_xticks(x, list("JFMAMJJASOND"))
        ax.set_xlabel("Month")
        ax.set_title(site)
        evi_ax.set_ylim(0.43, 0.57)
        evi_ax.set_yticks(np.arange(0.44, 0.561, 0.02))
        evi_ax.set_ylabel("EVI", color="green")
        evi_ax.tick_params(axis="y", colors="green")
        evi_ax.spines["right"].set_color("green")
        ax.text(0.04, 0.93,
                rf"$\Delta t$: {v.lag_mean_month.iloc[0]:.1f} $\pm$ {v.lag_spatial_sd_month.iloc[0]:.1f} month",
                transform=ax.transAxes, fontsize=9)
        ax.text(0.04, 0.83, f"MAP: {v.precipitation_mm.sum():.0f} mm/year",
                color="#0C7BDC", transform=ax.transAxes, fontsize=9)
    save_figure(fig, output, "Figure2_seasonality")


def map_panel(values, geo, locations, output, stem, label, is_lag=False):
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_axes([0.10, 0.16, 0.80, 0.76], projection=projection)
    if is_lag:
        cmap = mcolors.ListedColormap(plt.cm.RdBu_r(np.linspace(0.1, 0.8, 7)))
        norm = mcolors.BoundaryNorm(np.arange(8), 7)
        bins = np.arange(-0.5, 7.5, 1)
    else:
        # Match the supplied deciduousness-amplitude reference: BrBG_r with
        # a custom green-to-brown ramp and a fixed 0-50% display range.
        base_cmap = plt.cm.BrBG_r
        cmap = LinearSegmentedColormap.from_list(
            "custom_brbg",
            [(0 / 50, base_cmap(0.15)),
             (15 / 50, base_cmap(0.50)),
             (50 / 50, base_cmap(0.90))],
            N=256,
        )
        cmap = ListedColormap(cmap(np.linspace(0, 1, 256)))
        cmap.set_under(color=(0.83, 0.83, 0.83, 1.0))
        norm, bins = mcolors.Normalize(0, 50), 50
    height, width = values.shape
    extent = [geo[0], geo[0] + geo[1] * width, geo[3] + geo[5] * height, geo[3]]
    artist = ax.imshow(values, extent=extent, origin="upper", transform=projection,
                       cmap=cmap, norm=norm, interpolation="nearest")
    add_boundary(ax)
    for site, loc in locations.iterrows():
        ax.plot(loc.Lon, loc.Lat, marker="*", color="black", ms=9, transform=projection)
        ax.text(loc.Lon - 0.6, loc.Lat - 1.8, site, fontsize=9, transform=projection)
    ax.set_extent([-80, -44, -22, 10], crs=projection)
    ax.set_xticks(np.arange(-80, -40, 10), crs=projection)
    ax.set_yticks(np.arange(-20, 11, 10), crs=projection)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    # White inset panel keeps histogram labels clear of the basin map.
    backing = ax.inset_axes([0.65, 0.015, 0.35, 0.35])
    backing.set_facecolor("white")
    backing.set_xticks([])
    backing.set_yticks([])
    for spine in backing.spines.values():
        spine.set_visible(False)
    inset = ax.inset_axes([0.70, 0.11, 0.23, 0.20])
    valid = values[np.isfinite(values)]
    inset.hist(valid, bins=bins, weights=np.ones(len(valid)) / len(valid),
               color="grey", alpha=0.6, edgecolor="black", linewidth=0.25)
    inset.set_ylabel("Proportion", fontsize=8)
    inset.yaxis.tick_right()
    inset.yaxis.set_label_position("right")
    inset.set_xlabel("Lag (month)" if is_lag else "Amplitude (%)", fontsize=8)
    inset.tick_params(labelsize=7)
    if is_lag:
        inset.set_xlim(-0.6, 6.6)
        inset.set_xticks(np.arange(7))
        inset.set_ylim(0, 0.30)
    else:
        inset.set_xlim(0, 50)
    cax = fig.add_axes([0.15, 0.075, 0.70, 0.027])
    bar = fig.colorbar(artist, cax=cax, orientation="horizontal", drawedges=is_lag)
    if is_lag:
        bar.set_ticks(np.arange(7) + 0.5, labels=[str(i) for i in range(7)])
    bar.set_label(label)
    save_figure(fig, output, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figure2")
    parser.add_argument("--lag-path", type=Path, default=DATA / "seasonality/time_lag_map.tif")
    parser.add_argument("--error-bars", choices=("interannual", "spatial"), default="interannual")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == DATA or DATA in output.parents:
        raise ValueError("Outputs must not be written inside data/")
    if output.exists():
        raise FileExistsError("Choose a new --output-dir to preserve existing results")
    locations = pd.read_csv(DATA / "phenocam/ATTO_RJA_Location.csv").set_index("Site").loc[["ATTO", "RJA"]]
    mask, _ = read_raster(DATA / "forest_mask/MCD12Q1_Amazon.tif")
    forest = cv2.resize(mask, (786, 650), interpolation=cv2.INTER_NEAREST) == 2
    dec, geo = read_raster(DATA / "deciduousness/Composite_Data_5km_gf_3y.tif")
    dec[dec > 1000] = np.nan
    dec[~forest] = np.nan
    valid = np.isfinite(dec).any(axis=2)
    amplitude = np.full(forest.shape, np.nan, dtype=np.float32)
    amplitude[valid] = (np.nanmax(dec[valid], axis=1) - np.nanmin(dec[valid], axis=1)) / 10.0
    lag, lag_geo = read_raster(args.lag_path)
    if lag.shape != forest.shape or not np.allclose(geo, lag_geo, rtol=0, atol=1e-8):
        raise ValueError("Lag and deciduousness grids differ")
    if args.error_bars == "interannual":
        sites = figure2_site_data(args.lag_path).rename(columns={
            "deciduousness_interannual_sd": "deciduousness_sd", "evi_interannual_sd": "evi_sd"})
    else:
        sites = spatial_site_data(locations, forest, lag)
    sites["error_bar_kind"] = args.error_bars
    sites["sd_ddof"] = 0
    output.mkdir(parents=True)
    sites.to_csv(output / "Figure2_site_monthly.csv", index=False)
    np.savez_compressed(output / "Figure2_map_arrays.npz", amplitude_percent=amplitude, time_lag_month=lag)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 10, "pdf.fonttype": 42})
    seasonality(sites, output)
    map_panel(amplitude, geo, locations, output, "Figure2_deciduousness_amplitude",
              "Deciduousness amplitude (%)")
    map_panel(lag, lag_geo, locations, output, "Figure2_time_lag", r"$\Delta t$ (month)", True)
    report = dict(error_bars=args.error_bars, sd_ddof=0, lag_input=str(args.lag_path),
                  amplitude_valid_pixels=int(valid.sum()), lag_valid_pixels=int(np.isfinite(lag).sum()),
                  scope="Plots from released processed inputs; no upstream satellite retrieval.")
    (output / "Figure2_plot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote three Figure 2 plots (PNG/PDF), source arrays and site CSV to {output}")


if __name__ == "__main__":
    main()
