"""Generate only time_lag_map.tif from released processed inputs.

Cyclic Pearson correlation is retained solely to select the lag; no separate
correlation or p-value products are generated.
"""

# In[] Imports
import argparse
from pathlib import Path
import copy
import numpy as np
import cv2
from osgeo import gdal
import pandas as pd

# In[] Workflow
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "seasonality")
parser.add_argument("--forest-mask", type=Path, default=ROOT / "data" / "forest_mask" / "MCD12Q1_Amazon.tif",
                    help="The released coarse mask reproduces the manuscript lag raster; native/ is a different support.")
args = parser.parse_args()
OUTPUT_DIR = args.output_dir.resolve()
if OUTPUT_DIR == ROOT / "data" or ROOT / "data" in OUTPUT_DIR.parents:
    raise ValueError("Generated outputs must not overwrite released data/ inputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In[] Functions
def readTif_gdal(fileName, nbands=36):
    gdal.UseExceptions()
    dataset = gdal.Open(fileName)
    if dataset == None:
        raise FileNotFoundError(fileName)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    if im_data.ndim == 3:
        im_data = np.transpose(im_data, [1, 2, 0])
    return dataset.GetGeoTransform(), dataset.GetProjection(), im_data

def save_tif(grouthTif, savePath, Geo_, Projection_, nbands):
    gdal.UseExceptions()
    if Path(savePath).exists():
        raise FileExistsError(f"Output exists; choose a new --output-dir: {savePath}")
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    datatype = gdal.GDT_Float32

    outputData = driver.Create(savePath, grouthTif.shape[1], grouthTif.shape[0], nbands, datatype)

    outputData.SetGeoTransform(Geo_)
    outputData.SetProjection(Projection_)

    if nbands == 1:
        outputData.GetRasterBand(1).WriteArray(grouthTif)
        outputData.GetRasterBand(1).SetNoDataValue(np.nan)

    else:
        for i in range(nbands):
            outputData.GetRasterBand(i + 1).WriteArray(grouthTif[:, :, i])
            outputData.GetRasterBand(1).SetNoDataValue(np.nan)

    del outputData

raws_y, columns_x = 786, 650
cls_modis_path = str(args.forest_mask)

_, _, cls_md = readTif_gdal(cls_modis_path)
cls_md_forest = copy.deepcopy(cls_md)
cls_md_forest = cls_md_forest.astype(np.float32)
cls_md_forest[cls_md_forest != 2] = np.nan
cls_modis_coarse = cv2.resize(cls_md_forest, (raws_y, columns_x), interpolation=cv2.INTER_NEAREST)

forest_mask = ~np.isnan(cls_modis_coarse)

ind_where = np.where(forest_mask == 1)
print('evergreen forest pixels:', len(ind_where[0]))

raws_y, columns_x = 786, 650

EVI_path = str(ROOT / 'data/seasonality/BRDF_EVI_3years_mean.tif')

_, _, EVI_amazon = readTif_gdal(EVI_path)
EVI_amazon = cv2.resize(EVI_amazon, (raws_y, columns_x))
EVI_amazon = EVI_amazon

dec_path = str(ROOT / 'data/deciduousness/Composite_Data_5km_gf_3y.tif')
_geo, _prj, dec_month = readTif_gdal(dec_path)
dec_month = dec_month.astype(np.float32)
dec_month[dec_month > 1000] = np.nan

dec_month_res = dec_month.reshape(-1, 12)

forest_mask_res = forest_mask.reshape(-1)

dec_month_forest_res = dec_month_res[forest_mask_res]
evi_forest_res = EVI_amazon.reshape(-1, 12)[forest_mask_res]
time_lag = np.full(forest_mask_res.sum(), np.nan)

pair_valid = ~np.isnan(dec_month_forest_res) & ~np.isnan(evi_forest_res)
pair_count = pair_valid.sum(axis=1)
pair_dec_sum = np.nansum(np.where(pair_valid, dec_month_forest_res, np.nan), axis=1)
valid_rows = (pair_count >= 3) & (pair_dec_sum != 0)

# Fill each EVI series using the same cyclic interpolation as the original loop.
evi_repeat = np.concatenate([evi_forest_res, evi_forest_res], axis=1)
evi_repeat = pd.DataFrame(evi_repeat).interpolate(method='linear', axis=1).to_numpy()
evi_filled = evi_repeat[:, :12].copy()
missing_evi = np.isnan(evi_filled)
evi_filled[missing_evi] = evi_repeat[:, 12:24][missing_evi]

dec_repeated = np.concatenate([dec_month_forest_res, dec_month_forest_res], axis=1)
cor_by_lag = np.full((dec_repeated.shape[0], 12), np.nan, dtype=np.float32)
for shift_id in range(12):
    x_lag = dec_repeated[:, shift_id:shift_id + 12]
    lag_valid = ~np.isnan(x_lag) & ~np.isnan(evi_filled)
    x_mean = np.full((x_lag.shape[0], 1), np.nan, dtype=np.float32)
    y_mean = np.full((x_lag.shape[0], 1), np.nan, dtype=np.float32)
    has_pairs = lag_valid.any(axis=1)
    if np.any(has_pairs):
        x_mean[has_pairs] = np.nanmean(
            np.where(lag_valid[has_pairs], x_lag[has_pairs], np.nan),
            axis=1, keepdims=True)
        y_mean[has_pairs] = np.nanmean(
            np.where(lag_valid[has_pairs], evi_filled[has_pairs], np.nan),
            axis=1, keepdims=True)
    x_centered = np.where(lag_valid, x_lag - x_mean, np.nan)
    y_centered = np.where(lag_valid, evi_filled - y_mean, np.nan)
    numerator = np.nansum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.nansum(x_centered ** 2, axis=1) * np.nansum(y_centered ** 2, axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        cor_by_lag[:, shift_id] = numerator / denominator

cor_by_lag[~valid_rows, :] = np.nan
lag = np.argmax(np.where(np.isnan(cor_by_lag), -np.inf, cor_by_lag), axis=1)
time_lag[valid_rows] = lag[valid_rows]

time_lag_res = np.full(forest_mask_res.shape, np.nan)
time_lag_res[forest_mask_res] = time_lag
time_lag_map = time_lag_res.reshape(columns_x, raws_y)
time_lag_map[time_lag_map > 6] = 12 - time_lag_map[time_lag_map > 6]
save_tif(time_lag_map, str(OUTPUT_DIR / 'time_lag_map.tif'), _geo, _prj, 1)
print(f"Wrote time_lag_map.tif; valid pixels: {np.isfinite(time_lag_map).sum()}")
