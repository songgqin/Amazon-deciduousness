"""Verify the main-text Figure 4 basin-scale GPP-SIF result.

This is a repository-relative port of the original R4_2 QA calculation.  It
keeps the original target grid, forest-mask resize rule, per-pixel
normalization, six-month validity rule, and formulation-balanced ensemble.
The 79.9% value is a QA gate: it is never used to alter the calculation.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from osgeo import gdal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "qa"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_WIDTH = 786
TARGET_HEIGHT = 650
MIN_VALID_MONTHS = 6
EXPECTED_ORIGINAL_ROUNDED_PERCENT = 79.9

FOREST_MASK = DATA / "forest_mask" / "MCD12Q1_Amazon.tif"
SIF_PATHS = [
    DATA / "gpp" / "sif" / "Amazon_GOSIF_mean_calibrated.tif",
    DATA / "gpp" / "sif" / "CSIF_Amazon_005_calibrated.tif",
]

CONVENTIONAL_PATHS = {
    "EC-LUE": [
        DATA / "gpp" / "outputs" / "EC_LUE_EVI_GPP_local.tif",
        DATA / "gpp" / "outputs" / "EC_LUE_kNDVI_GPP_local.tif",
        DATA / "gpp" / "outputs" / "EC_LUE_NDVI_GPP_local.tif",
        DATA / "gpp" / "outputs" / "EC_LUE_MODIS_LAI_GPP_local.tif",
    ],
    "MOD-LUE": [DATA / "gpp" / "outputs" / "MOD_LUE_GPP_local.tif"],
    "TL-EC": [DATA / "gpp" / "outputs" / "TL_EC_LUE_GPP_local.tif"],
}

LD_PATHS = {
    "EC-LUE": DATA / "gpp" / "outputs" / "EC_LUE_LAI_Dec_Demography_GPP.tif",
    "MOD-LUE": DATA / "gpp" / "outputs" / "MOD_LUE_LAI_Dec_Demography_GPP.tif",
    "TL-EC": DATA / "gpp" / "outputs" / "TL_EC_LUE_LAI_Dec_Demography_GPP.tif",
}


def read_raster(path: Path) -> np.ndarray:
    """Read a GDAL raster as [row, column, band] and normalize common nodata."""

    dataset = gdal.Open(str(path))
    if dataset is None:
        raise FileNotFoundError(path)
    array = dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)
    if array.ndim == 3 and array.shape[0] <= 36:
        array = np.transpose(array, (1, 2, 0))
    array = array.astype(np.float32, copy=False)
    array = array.copy()
    array[(array == 65535) | (array == -9999)] = np.nan
    return array


def read_resized(path: Path) -> np.ndarray:
    return cv2.resize(
        read_raster(path),
        (TARGET_WIDTH, TARGET_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)


def normalize_ts(values: np.ndarray) -> np.ndarray:
    minimum = np.nanmin(values, axis=1, keepdims=True)
    maximum = np.nanmax(values, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (values - minimum) / (maximum - minimum)


def vectorized_pearsonr(values_x: np.ndarray, values_y: np.ndarray) -> np.ndarray:
    joint_valid = np.isfinite(values_x) & np.isfinite(values_y)
    x_safe = np.where(joint_valid, values_x, np.nan)
    y_safe = np.where(joint_valid, values_y, np.nan)
    x_centered = x_safe - np.nanmean(x_safe, axis=1, keepdims=True)
    y_centered = y_safe - np.nanmean(y_safe, axis=1, keepdims=True)
    numerator = np.nansum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(
        np.nansum(x_centered**2, axis=1)
        * np.nansum(y_centered**2, axis=1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def delta_r_map(
    ld_gpp: np.ndarray,
    conventional_gpp: list[np.ndarray],
    sif_reference: np.ndarray,
    forest_mask: np.ndarray,
) -> np.ndarray:
    """Return the formulation-balanced Delta-r map from the original QA code."""

    ld_ts = ld_gpp[forest_mask]
    sif_ts = sif_reference[forest_mask]
    formulation_maps = []

    for conventional in conventional_gpp:
        conventional_ts = conventional[forest_mask]
        valid = (
            (np.sum(np.isfinite(sif_ts), axis=1) >= MIN_VALID_MONTHS)
            & (np.sum(np.isfinite(ld_ts), axis=1) >= MIN_VALID_MONTHS)
            & (np.sum(np.isfinite(conventional_ts), axis=1) >= MIN_VALID_MONTHS)
        )
        values = np.full(np.sum(forest_mask), np.nan, dtype=np.float32)
        if np.any(valid):
            values[valid] = (
                vectorized_pearsonr(ld_ts[valid], sif_ts[valid])
                - vectorized_pearsonr(conventional_ts[valid], sif_ts[valid])
            )
        output = np.full(forest_mask.shape, np.nan, dtype=np.float32)
        output[forest_mask] = values
        formulation_maps.append(output)

    return np.nanmean(np.stack(formulation_maps, axis=0), axis=0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_paths() -> list[Path]:
    paths = [FOREST_MASK, *SIF_PATHS]
    paths.extend(path for values in CONVENTIONAL_PATHS.values() for path in values)
    paths.extend(LD_PATHS.values())
    return paths


def run() -> pd.DataFrame:
    missing = [path for path in input_paths() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 4 input(s):\n" + "\n".join(map(str, missing)))

    print(">>> Reading the original forest mask and SIF references...")
    forest_class = read_raster(FOREST_MASK).copy()
    forest_class[forest_class != 2] = -1
    forest_mask = cv2.resize(
        forest_class,
        (TARGET_WIDTH, TARGET_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    ) == 2

    sif_1 = read_resized(SIF_PATHS[0]) * 0.0001
    sif_2 = read_resized(SIF_PATHS[1])
    sif_1[sif_1 < 0] = np.nan
    sif_2[sif_2 < 0] = np.nan
    sif_reference = np.full_like(sif_1, np.nan, dtype=np.float32)
    sif_reference[forest_mask] = np.nanmean(
        np.stack(
            [
                normalize_ts(sif_1[forest_mask]),
                normalize_ts(sif_2[forest_mask]),
            ],
            axis=0,
        ),
        axis=0,
    )

    conventional = {
        formulation: [read_resized(path) for path in paths]
        for formulation, paths in CONVENTIONAL_PATHS.items()
    }

    formulation_maps = []
    formulation_summary = {}
    for formulation in ("EC-LUE", "MOD-LUE", "TL-EC"):
        print(f">>> Calculating formulation-balanced Delta r: {formulation}")
        delta = delta_r_map(
            read_resized(LD_PATHS[formulation]),
            conventional[formulation],
            sif_reference,
            forest_mask,
        )
        formulation_maps.append(delta)
        valid = np.isfinite(delta)
        formulation_summary[formulation] = {
            "valid_forest_pixels": int(np.sum(valid)),
            "improved_forest_pixels": int(np.sum(delta[valid] > 0)),
            "improved_forest_pixels_percent": float(np.mean(delta[valid] > 0) * 100),
        }

    ensemble = np.nanmean(np.stack(formulation_maps, axis=0), axis=0)
    valid = forest_mask & np.isfinite(ensemble)
    improved = ensemble[valid] > 0
    original_percent = float(np.mean(improved) * 100)
    summary = pd.DataFrame(
        [
            {
                "Scenario": "Original Eq. 9",
                "Own_Valid_Forest_Pixels": int(np.sum(valid)),
                "Own_Improved_Forest_Pixels": int(np.sum(improved)),
                "Own_Improved_Forest_Pixels_Percent": original_percent,
                "Expected_Rounded_Percent": EXPECTED_ORIGINAL_ROUNDED_PERCENT,
                "Pass": round(original_percent, 1) == EXPECTED_ORIGINAL_ROUNDED_PERCENT,
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "maintext_fig4_pixel_proportion.csv", index=False)

    metadata = {
        "workflow": "R42_04b_Basin_GPP_SIF_Pixel_Proportion",
        "definition": "Unweighted forest-pixel proportion with Delta r > 0.",
        "delta_r_definition": "r(LD-LUE, normalized GOSIF-CSIF reference) minus r(conventional LUE, the same reference).",
        "target_grid": [TARGET_WIDTH, TARGET_HEIGHT],
        "minimum_valid_months": MIN_VALID_MONTHS,
        "original_improved_forest_pixels_percent": original_percent,
        "original_rounds_to_79_9": round(original_percent, 1) == EXPECTED_ORIGINAL_ROUNDED_PERCENT,
        "formulation_summary": formulation_summary,
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in input_paths()},
    }
    with (OUTPUT_DIR / "maintext_fig4_qa.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    print(summary.to_string(index=False))
    print(f"Saved: {OUTPUT_DIR / 'maintext_fig4_pixel_proportion.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'maintext_fig4_qa.json'}")
    if not metadata["original_rounds_to_79_9"]:
        raise RuntimeError(
            f"Figure 4 QA failed: computed {original_percent:.6f}% instead of "
            f"the local main-text value rounded to {EXPECTED_ORIGINAL_ROUNDED_PERCENT:.1f}%."
        )
    return summary


if __name__ == "__main__":
    run()
