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

rainfall_path = r'data/climate/precipiation.tif'
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

for i in tqdm(np.arange(0, dec_month_forest_res.shape[0])):

    x_raw_list = ~np.isnan(dec_month_forest_res[i]) & ~np.isnan(evi_forest_res[i])

    x_raw_month = x_list[x_raw_list]

    dec_list = dec_month_forest_res[i][x_raw_list]
    evi_list = evi_forest_res[i][x_raw_list]

    if np.sum(dec_list) == 0 or len(dec_list)<3:
        continue

    cor_forest_res[i] = np.corrcoef(dec_list, evi_list)[0, 1]

    x = copy.deepcopy(dec_month_forest_res[i])
    y = copy.deepcopy(evi_forest_res[i])

    if np.sum(np.isnan(y)) > 0:
        y_repeat = np.append(y, y, axis=0)
        y_repeat = pd.DataFrame(y_repeat).interpolate(method='linear').to_numpy().reshape(-1)
        y = y_repeat[:12]
        y[np.isnan(y)] = y_repeat[12:24][np.isnan(y)]

    cor_list = []
    dec_repeated = np.append(x, x, axis=0)

    p_value_list = []

    for shift_id in range(12):
        cor_list.append(np.corrcoef(dec_repeated[shift_id:shift_id + 12], y)[0, 1])
        p_value_list.append(stats.pearsonr(dec_repeated[shift_id:shift_id + 12], y)[1])

    lag = np.argmax(np.array(cor_list))

    p_value_forest_res[i] = p_value_list[lag]

    time_lag[i] = lag

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

save_path = r'data/seasonality/time_lag_map_0521.tif'

save_tif(time_lag_map2, save_path, _geo, _prj, 1)
