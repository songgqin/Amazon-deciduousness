"""Figure 2 correlation and time-lag raster generation.

"""

# In[] Imports
import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from osgeo import gdal
import matplotlib
from tqdm import tqdm
import seaborn as sns
from scipy.signal import find_peaks
from scipy import stats
import pandas as pd
import copy

from skimage import morphology
import seaborn as sns

# In[] Workflow
matplotlib.use("Agg")
matplotlib.rcParams['figure.dpi'] = 200
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# In[] Functions
def readTif_gdal(fileName, nbands=36):
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    dataset = gdal.Open(fileName)
    if dataset == None:
        print("cannot open file:" + fileName)
        return
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    if im_data.shape[0] <= nbands:
        im_data = np.transpose(im_data, [1, 2, 0])
    return dataset.GetGeoTransform(), dataset.GetProjection(), im_data

def save_tif(grouthTif, savePath, Geo_, Projection_, nbands):
    gdal.UseExceptions()
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
cls_modis_path = r'data/forest_mask/MCD12Q1_Amazon.tif'

_, _, cls_md = readTif_gdal(cls_modis_path)
cls_md_forest = copy.deepcopy(cls_md)
cls_md_forest = cls_md_forest.astype(np.float32)
cls_md_forest[cls_md_forest != 2] = np.nan
cls_modis_coarse = cv2.resize(cls_md_forest, (raws_y, columns_x), interpolation=cv2.INTER_NEAREST)

forest_mask = ~np.isnan(cls_modis_coarse)

ind_where = np.where(forest_mask == 1)
print('evergreen forest pixels:', len(ind_where[0]))

raws_y, columns_x = 786, 650

EVI_path = r'data/seasonality/BRDF_EVI_3years_mean.tif'

_, _, EVI_amazon = readTif_gdal(EVI_path)
EVI_amazon = cv2.resize(EVI_amazon, (raws_y, columns_x))
EVI_amazon = EVI_amazon

rainfall_path = r'data/drivers/inputs/hydroclimate_precipitation_ERA.tif'
_geo, _prj, rainfall = readTif_gdal(rainfall_path)
rainfall = cv2.resize(rainfall, (raws_y, columns_x))

par_path = r'data/climate/climate_par.tif'
_geo, _prj, par = readTif_gdal(par_path)
par = cv2.resize(par, (raws_y, columns_x))

dec_path = r'data/deciduousness/Composite_Data_5km_gf_3y.tif'
_geo, _prj, dec_month = readTif_gdal(dec_path)
dec_month = dec_month.astype(np.float32)
dec_month[dec_month > 1000] = np.nan

dec_month_res = dec_month.reshape(-1, 12)

forest_mask_res = forest_mask.reshape(-1)

cor_np = np.ones((forest_mask_res.shape[0])) * np.nan
cor_rain_np = np.ones((forest_mask_res.shape[0])) * np.nan
cor_par_np = np.ones((forest_mask_res.shape[0])) * np.nan

pcor_par_np = np.ones((forest_mask_res.shape[0])) * np.nan
pcor_rain_np = np.ones((forest_mask_res.shape[0])) * np.nan

cor_evi_rain = np.ones((forest_mask_res.shape[0])) * np.nan
cor_evi_par = np.ones((forest_mask_res.shape[0])) * np.nan

rainfall_res = rainfall.reshape(-1, 12)
evi_res = EVI_amazon.reshape(-1, 12)
par_res = par.reshape(-1, 12)

rainfall_forest_res = rainfall_res[forest_mask_res]
dec_month_forest_res = dec_month_res[forest_mask_res]
evi_forest_res = evi_res[forest_mask_res]
par_forest_res = par_res[forest_mask_res]

cor_forest_res = cor_np[forest_mask_res]
cor_rain_forest_res = cor_rain_np[forest_mask_res]
cor_par_forest_res = cor_par_np[forest_mask_res]

pcor_par_forest_res = copy.deepcopy(cor_par_forest_res)
pcor_rain_forest_res = copy.deepcopy(cor_rain_forest_res)

cor_evi_rain_forest_res = cor_evi_rain[forest_mask_res]
cor_evi_par_forest_res = cor_evi_par[forest_mask_res]

p_value_forest_res = copy.deepcopy(cor_forest_res) * np.nan

x_list = np.arange(12)
month_list = np.arange(1, 13)
time_lag = copy.deepcopy(cor_par_forest_res) * np.nan

pair_valid = ~np.isnan(dec_month_forest_res) & ~np.isnan(evi_forest_res)
pair_count = pair_valid.sum(axis=1)
pair_dec_sum = np.nansum(np.where(pair_valid, dec_month_forest_res, np.nan), axis=1)
valid_rows = (pair_count >= 3) & (pair_dec_sum != 0)

dec_pair = np.where(pair_valid, dec_month_forest_res, np.nan)
evi_pair = np.where(pair_valid, evi_forest_res, np.nan)
dec_centered = dec_pair - np.nanmean(dec_pair, axis=1, keepdims=True)
evi_centered = evi_pair - np.nanmean(evi_pair, axis=1, keepdims=True)
cor_num = np.nansum(dec_centered * evi_centered, axis=1)
cor_den = np.sqrt(np.nansum(dec_centered ** 2, axis=1) * np.nansum(evi_centered ** 2, axis=1))
with np.errstate(divide='ignore', invalid='ignore'):
    cor_forest_res[valid_rows] = cor_num[valid_rows] / cor_den[valid_rows]

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
    x_mean = np.nanmean(np.where(lag_valid, x_lag, np.nan), axis=1, keepdims=True)
    y_mean = np.nanmean(np.where(lag_valid, evi_filled, np.nan), axis=1, keepdims=True)
    x_centered = np.where(lag_valid, x_lag - x_mean, np.nan)
    y_centered = np.where(lag_valid, evi_filled - y_mean, np.nan)
    numerator = np.nansum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.nansum(x_centered ** 2, axis=1) * np.nansum(y_centered ** 2, axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        cor_by_lag[:, shift_id] = numerator / denominator

cor_by_lag[~valid_rows, :] = np.nan
lag = np.argmax(np.where(np.isnan(cor_by_lag), -np.inf, cor_by_lag), axis=1)
time_lag[valid_rows] = lag[valid_rows]

cor_np[forest_mask_res] = cor_forest_res
cor_np_map = cor_np.reshape(columns_x, raws_y)

time_lag_res = copy.deepcopy(cor_np) * np.nan
time_lag_res[forest_mask_res] = time_lag
time_lag_map = time_lag_res.reshape(columns_x, raws_y)

p_value_np = copy.deepcopy(cor_np) * np.nan
p_value_np[forest_mask_res] = p_value_forest_res
p_value_map = p_value_np.reshape(columns_x, raws_y)

time_lag_map2 = copy.deepcopy(time_lag_map)

time_lag_map2[time_lag_map2 > 6] = 12 - time_lag_map2[time_lag_map2 > 6]

cor_np_map[~forest_mask] = np.nan

os.makedirs(r'outputs/seasonality', exist_ok=True)
save_path = r'outputs/seasonality/time_lag_map_0521.tif'

save_tif(time_lag_map2, save_path, _geo, _prj, 1)

correlation_path = r'outputs/seasonality/Cor_Dec_EVI_0521.tif'
save_tif(cor_np_map, correlation_path, _geo, _prj, 1)
