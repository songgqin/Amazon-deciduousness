"""Unmix the released RJA Sentinel-2 scenes and summarize seasonality."""

import argparse
from pathlib import Path

# Load GDAL before TensorFlow on Windows.
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_preprocessing import save_tif_from_valid_pixels
from ecae_modules_four_bands import get_endmember_band_range_percentage, plot_endmember, train_autoencoder
import unmixing


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data/sentinel2/rja_s2"
BAND_NUMBERS = [1, 2, 3, 7]
ENDMEMBERS = ["shade", "deciduous", "evergreen"]
MIN_TRAINING_EVI = 0.15
DEFAULT_INITIALIZATION_SEED = 42


def get_abundance_map(image, endmember_mean):
    flat = image.reshape(-1, image.shape[-1])
    valid = np.all(flat != 0, axis=1)
    result = np.zeros((len(flat), endmember_mean.shape[1]))
    if valid.any():
        result[valid] = unmixing.sunsal(
            endmember_mean, flat[valid].T, al_iters=200, positivity=True,
            addone=True, verbose=False,
        )[0].T
    return result.reshape((*image.shape[:-1], endmember_mean.shape[1]))


def read_scene(path):
    dataset = gdal.Open(str(path))
    if dataset.RasterCount < max(BAND_NUMBERS):
        raise ValueError(f"Expected at least seven bands in {path}")
    image = dataset.ReadAsArray(band_list=BAND_NUMBERS).astype(np.float32)
    image = np.nan_to_num(image)
    image[(image < 0) | (image > 1)] = 0
    return image, dataset.GetGeoTransform(), dataset.GetProjection()


def get_training_pixels(image):
    blue, _, red, nir = image
    denominator = nir + 6 * red - 7.5 * blue + 1
    evi = np.divide(
        2.5 * (nir - red), denominator, out=np.zeros_like(nir, dtype=np.float64),
        where=denominator != 0,
    )
    nonzero = np.any(image != 0, axis=0)
    eligible = nonzero & (evi >= MIN_TRAINING_EVI)
    pixels = image[:, eligible].T
    if len(pixels) >= 2:
        _, pixels = train_test_split(pixels, test_size=0.8, random_state=42)
    return pixels, int(nonzero.sum()), int(eligible.sum())


def fit_year(scenes, initialization_seed):
    samples, counts = [], []
    for relative_path in scenes.relative_path:
        image, _, _ = read_scene(INPUT_DIR / relative_path)
        pixels, nonzero, eligible = get_training_pixels(image)
        if len(pixels):
            samples.append(pixels)
        counts.append(dict(relative_path=relative_path, nonzero_pixels=nonzero,
                           evi_pass_pixels=eligible, train_pixels=len(pixels)))
    if not samples:
        raise ValueError("No eligible training pixels")
    training = np.concatenate(samples, axis=0)
    del samples
    ranges = get_endmember_band_range_percentage(training, percentile=95, percent=10)
    initial = np.asarray(ranges)[:, 2, :].T
    trained = train_autoencoder(
        Img=training, Img_clean=training, endnum=3, endmember_list=ranges, epoch=5,
        learning_rate=1e-3, batch_size=512, require_improvement=1,
        delta_increasing=1e-4, begin_with_zero=False, pre_initial=True, verbose=True,
        initialization_seed=initialization_seed,
    )
    return trained, initial, counts


def community_counts(values):
    valid = np.isfinite(values).all(axis=1) & np.any(values != 0, axis=1)
    classes = np.argmax(np.clip(values[valid], 0, 1), axis=1)
    shade, deciduous, evergreen = np.bincount(classes, minlength=3)
    vegetation = deciduous + evergreen
    return dict(shade_dominant_pixels=shade, deciduous_pixels=deciduous,
                evergreen_pixels=evergreen,
                deciduousness_pct=100 * deciduous / vegetation if vegetation else np.nan)


def write_seasonality(rows, output_dir):
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "scene_deciduousness.csv", index=False)
    months = pd.RangeIndex(1, 13, name="month")
    monthly = pd.DataFrame(index=months)
    for tag, data in [("all", table), ("legacy_comparison", table[table.legacy_comparison == 1])]:
        by_year = data.groupby(["year", "month"]).deciduousness_pct.mean()
        monthly[tag + "_pct"] = by_year.groupby("month").mean()
        monthly[tag + "_scenes"] = data.groupby("month").deciduousness_pct.count().reindex(months, fill_value=0)
        monthly[tag + "_years"] = by_year.groupby("month").count().reindex(months, fill_value=0)
    monthly.to_csv(output_dir / "monthly_seasonality.csv")
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    ax.plot(months, monthly.all_pct, "s--", color="#b95029", label="All supplied scenes")
    ax.plot(months, monthly.legacy_comparison_pct, "o-", color="#236a63", label="Legacy comparison dates")
    ax.set(xlabel="Month", ylabel="Deciduous-dominant pixels (%)",
           xticks=months, xticklabels="Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())
    ax.legend(frameon=False)
    fig.savefig(output_dir / "RJA_seasonality.png", dpi=200)
    plt.close(fig)


def sentinel_2_multiyears_unmixing(
    output_dir, use_reference_endmembers=False,
    initialization_seed=DEFAULT_INITIALIZATION_SEED,
):
    gdal.UseExceptions()
    scenes = pd.read_csv(INPUT_DIR / "scene_inventory.csv", parse_dates=["date"])
    scenes = scenes.sort_values(["date", "relative_path"])
    if scenes.empty or scenes.date.duplicated().any():
        raise ValueError("Expected a nonempty scene inventory with one scene per date")
    missing = [p for p in scenes.relative_path if not (INPUT_DIR / p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing scene {missing[0]}; run git lfs pull")
    reference = (pd.read_csv(INPUT_DIR / "reference_endmembers.csv", float_precision="round_trip")
                 if use_reference_endmembers else None)
    output_dir.mkdir(parents=True, exist_ok=False)
    spectral_rows, training_rows, summaries, scene_rows = [], [], [], []
    for year, group in scenes.groupby(scenes.date.dt.year):
        folder = output_dir / str(year)
        folder.mkdir()
        if reference is None:
            matrix, initial, counts = fit_year(group, initialization_seed)
            training_rows.extend(counts)
            outcome = "matches_initialization" if np.array_equal(matrix, initial) else "differs_from_initialization"
            plot_endmember(initial, f"RJA {year} initial endmembers")
            plt.savefig(folder / f"RJA_{year}_Index_Endmember.png", dpi=300)
            plt.close()
        else:
            year_reference = reference[reference.year == year].sort_values("band_index")
            if year_reference.band_index.tolist() != list(range(4)):
                raise ValueError(f"Missing reference endmember bands for {year}")
            dtypes = year_reference.dtype.unique().tolist()
            if dtypes not in (["float32"], ["float64"]):
                raise ValueError(f"Invalid reference matrix dtype for {year}")
            matrix = year_reference[ENDMEMBERS].to_numpy(dtype=dtypes[0])
            outcome = "reference_replay"
        if matrix.shape != (4, 3) or not np.isfinite(matrix).all() or not np.all(np.diff(matrix[3]) > 0):
            raise ValueError(f"Invalid or unordered endmembers for {year}")
        spectral_rows.extend(dict(year=year, band_index=band, dtype=matrix.dtype.name, **dict(zip(ENDMEMBERS, row)))
                             for band, row in enumerate(matrix))
        summaries.append(dict(year=year, scenes=len(group), outcome=outcome))
        if reference is None:
            summaries[-1]["initialization_seed"] = initialization_seed
        print(f"RJA {year}: {len(group)} scenes; {outcome}", flush=True)
        plot_endmember(matrix, f"RJA {year} endmembers")
        plt.savefig(folder / f"RJA_{year}_Endmember.png", dpi=300)
        plt.close()
        for scene in group.itertuples():
            path = INPUT_DIR / scene.relative_path
            image, geotransform, projection = read_scene(path)
            mask = np.any(image != 0, axis=0)
            values = get_abundance_map(image[:, mask].T, matrix).astype(np.float32)
            save_tif_from_valid_pixels(values, mask, str(folder / (path.stem + "_abu.tif")),
                                       geotransform, projection)
            scene_rows.append(dict(date=scene.date.date(), year=year, month=scene.date.month,
                                   relative_path=scene.relative_path,
                                   legacy_comparison=scene.legacy_comparison,
                                   **community_counts(values)))
    pd.DataFrame(spectral_rows).to_csv(output_dir / "endmembers.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "training_summary.csv", index=False)
    if training_rows:
        pd.DataFrame(training_rows).to_csv(output_dir / "training_pixels.csv", index=False)
    write_seasonality(scene_rows, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/sentinel2_unmixing/rja_s2")
    parser.add_argument("--use-reference-endmembers", action="store_true",
                        help="Use the supplied endmember matrices instead of refitting.")
    parser.add_argument("--initialization-seed", type=int, default=DEFAULT_INITIALIZATION_SEED,
                        help="TensorFlow initialization seed for refits (default: %(default)s).")
    args = parser.parse_args()
    sentinel_2_multiyears_unmixing(
        args.output_dir, args.use_reference_endmembers, args.initialization_seed,
    )
