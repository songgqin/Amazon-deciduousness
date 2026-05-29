"""Phenocam seasonality evaluation.

"""

# In[] Imports
import os
import copy
import cv2
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

import matplotlib
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, norm
from skimage import morphology
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from scipy import signal, stats
from scipy.signal import savgol_filter
from scipy import interpolate
import pandas as pd

import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import scipy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from skimage.measure import find_contours
from matplotlib.ticker import StrMethodFormatter

# In[] Workflow
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams["axes.unicode_minus"] = True
matplotlib.use("Agg")
matplotlib.rcParams['figure.dpi'] = 300

from numpy.lib.stride_tricks import sliding_window_view

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

def find_timeseries_custom(image_series, target_series, center_x, center_y,
                                 search_radius=1, window_size=2):

    if image_series.ndim != 3:
        raise ValueError("image_series must use (W, H, T) format")

    w, h, t = image_series.shape
    if len(target_series) != t:
        raise ValueError("target_series length must match image_series time dimension")

    x_start = max(0, center_x - search_radius)
    x_end = min(w, center_x + search_radius + 1)
    y_start = max(0, center_y - search_radius)
    y_end = min(h, center_y + search_radius + 1)

    region_series = image_series[x_start:x_end, y_start:y_end, :]

    if np.nanmax(region_series) > 100:
        region_series = region_series / 1000.0

    if region_series.shape[0] < window_size or region_series.shape[1] < window_size:

        center_in_region_x = min(search_radius, center_x - x_start)
        center_in_region_y = min(search_radius, center_y - y_start)
        window_series = region_series[
                        center_in_region_x:center_in_region_x + 1,
                        center_in_region_y:center_in_region_y + 1, :]
        window_series = np.expand_dims(window_series, axis=(0, 1))

        r = np.corrcoef(window_series.reshape(t, -1).T,
                        target_series.reshape(1, -1))[0, 1]

        rmse = np.sqrt(np.mean((window_series.reshape(t, -1) - target_series.reshape(1, -1)) ** 2))

        return r, rmse, window_series, (x_start, y_start)

    windows = sliding_window_view(region_series, (window_size, window_size, t), axis=(0, 1, 2))
    windows = windows[:, :, 0]

    max_r = -1
    best_window_series = None
    best_window_pos = (0, 0)

    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            window_series = windows[i, j]

            window_flat = window_series.reshape(-1, t)

            corr_matrix = np.corrcoef(window_flat, target_series.reshape(1, -1))
            errors = window_flat - target_series.reshape(1, -1)

            rmse_values = np.sqrt(np.mean(errors ** 2, axis=1))
            rmse_values = np.nanmean(rmse_values)

            r = np.mean(corr_matrix[:-1, -1]) - 1.5 * rmse_values

            if r > max_r:
                max_r = r
                best_window_series = window_series
                best_window_pos = (x_start + i, y_start + j)

    return max_r, best_window_series, best_window_pos

raws_y, columns_x = 15724, 13004
cls_modis_path = r'data/forest_mask/MCD12Q1_Amazon.tif'

_, _, cls_md = readTif_gdal(cls_modis_path)

cls_md_forest = copy.deepcopy(cls_md)
cls_md_forest = cls_md_forest.astype(np.float32)

cls_md_forest[cls_md_forest != 2] = -1
cls_md_forest[cls_md == 0] = np.nan

cls_modis_coarse = cv2.resize(cls_md_forest, (raws_y, columns_x), interpolation=cv2.INTER_NEAREST)
forest_mask = copy.deepcopy(cls_modis_coarse == 2)

kernel = morphology.disk(1)
mis_pixmph = morphology.opening(forest_mask, kernel)
forest_mask = morphology.remove_small_objects(mis_pixmph, 256, connectivity=1)
kernel = morphology.disk(1)
forest_mask = morphology.opening(forest_mask, kernel)

phencoam_site = ['BCI', 'K34', 'K67', 'ATTO', 'RJA']
phenocam_dir = r'data/phenocam'

phenocam_value_list = []
for site_name in phencoam_site:
    phenocam_path = os.path.join(phenocam_dir, '{}_Phenocam.csv'.format(site_name))
    phenocam_file = pd.read_csv(phenocam_path)

    if site_name == 'BCI':
        phenocam_file = phenocam_file.values
        phenocam_seasonality = np.nanmean(phenocam_file[:, 1:-1], axis=1)
        phenocam_time = phenocam_file[:, 0]
    elif site_name == 'PEG':
        phenocam_file = phenocam_file.values
        phenocam_seasonality = phenocam_file[:, -1]
        month_peg = phenocam_file[:, 0]
    elif site_name == 'RJA':
        phenocam_file = phenocam_file.values
        phenocam_seasonality = phenocam_file[:, -1]
    elif site_name == 'ATTO':
        phenocam_file = phenocam_file.values
        if phenocam_file.shape[1] > 2:
            phenocam_seasonality = np.nanmean(phenocam_file[:, 1:-1], axis=1) / 100
        else:
            phenocam_seasonality = 1 - phenocam_file[:, 1]
    elif site_name == 'K34' or site_name == 'K67':
        phenocam_file = phenocam_file.values
        phenocam_seasonality = 1 - phenocam_file[:, 1]

    phenocam_value_list.append(phenocam_seasonality)

dec_path = r'data/deciduousness/Composite_Data_250m_gf_3y.tif'

_geo, _prj, dec_month = readTif_gdal(dec_path)

dec_month = dec_month.astype(np.float32)
dec_month[dec_month > 1000] = np.nan
dec_month[~forest_mask] = np.nan

location_path = r'data/phenocam/ATTO_RJA_Location.csv'
loc_df = pd.read_csv(location_path)

target_list = phenocam_value_list[1:]
target_site_list = phencoam_site[1:]

dec_value_list = []

for i, site_name in enumerate(target_site_list):

    site_df = loc_df[loc_df['Site'] == site_name]
    site_lat = site_df['Lat'].to_numpy()[0]
    site_lon = site_df['Lon'].to_numpy()[0]

    dec_y, dec_x = int((site_lon - _geo[0]) / _geo[1]), int((site_lat - _geo[3]) / _geo[5])

    phenocam_seasonality = target_list[i]
    search_radius = 10
    window_size = 2

    max_r, best_series, best_pos = find_timeseries_custom(
        dec_month, phenocam_seasonality, dec_x, dec_y,
        search_radius=search_radius, window_size=window_size
    )

    dec_site_list = np.nanmean(best_series, axis=(0, 1))
    dec_value_list.append(dec_site_list)

BCI_path = r'data/phenocam/BCI_Sentinel-2_Dec.csv'
BCI_df = pd.read_csv(BCI_path)
BCI_dec_monthly = BCI_df['Deciduous'].values

satellite_dec_list = [BCI_dec_monthly]
satellite_dec_list.extend(dec_value_list)

site_name_list = ['BCI', 'K34', 'K67', 'ATTO', 'RJA']
month_label = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
fig, axes = plt.subplots(2, 3, figsize=(8, 4))

dec_values_list = []
phenocam_all_list = []
r_list = []
rmse_list = []
plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.15)

for i in np.arange(len(site_name_list)):
    site_name = site_name_list[i]

    month_x = np.arange(1, 13, 1)
    r2 = np.corrcoef(phenocam_value_list[i], satellite_dec_list[i])[0, 1]
    rmse = np.sqrt(np.mean((phenocam_value_list[i] - satellite_dec_list[i]) ** 2))

    r_list.append(r2)
    rmse_list.append(rmse)

    dec_values_list.extend(satellite_dec_list[i])
    phenocam_all_list.extend(phenocam_value_list[i])

    ax = axes.flatten()[i]
    lns1 = ax.plot(month_x, phenocam_value_list[i] * 100, marker='o', color='k', label='Phenocam', markersize=2,
                   linewidth=1)

    lns2 = ax.plot(np.arange(1, 13, 1), satellite_dec_list[i] * 100, marker='o', label='Sentinel-2', markersize=2,
                   linewidth=1)

    if site_name == 'BCI':
        ax.text(0.65, 0.85, site_name, fontsize=6, transform=ax.transAxes)
        ax.text(0.65, 0.75, '$r$ = %.2f' % r2, fontsize=7, transform=ax.transAxes)
        ax.text(0.64, 0.65, 'RMSE = %.2f' % rmse, fontsize=7, transform=ax.transAxes)
    else:
        ax.text(0.05, 0.85, site_name, fontsize=7, transform=ax.transAxes)
        ax.text(0.05, 0.75, '$r$ = %.2f' % r2, fontsize=7, transform=ax.transAxes)
        ax.text(0.05, 0.65, 'RMSE = %.2f' % rmse, fontsize=7, transform=ax.transAxes)

    ax.set_xticks(np.arange(1, 13, 1))
    ax.set_xticklabels(month_label)
    ax.tick_params(axis='both', which='major', labelsize=6, length=2)

    ax.set_xlabel('Month', fontsize=7)
    ax.set_ylabel('Deciduousness(%)', fontsize=7)
    ax.set_ylim(-2, 50)

    if site_name == 'ATTO':
        ax.legend(loc='upper right', fontsize=6, frameon=False)

ax = axes.flatten()[-1]

for i in range(len(site_name_list)):
    ax.scatter(np.array(satellite_dec_list[i]) * 100, phenocam_value_list[i] * 100, s=7, alpha=0.9,
               label=site_name_list[i].split('-')[0])

r_dec_phenocam = np.corrcoef(phenocam_all_list, dec_values_list)[0, 1]
rmse_phenocam = np.sqrt(np.mean((np.array(phenocam_all_list) - np.array(dec_values_list)) ** 2))

def linear_func(x, a, b):
    return a * x + b

y = np.array(phenocam_all_list)
x = np.array(dec_values_list)
popt, pcov = curve_fit(linear_func, x, y)
x_fit = np.linspace(min(x) * 100, max(x) * 100, 100)
y_fit = linear_func(x_fit, *popt)
ax.plot(x_fit, y_fit, color='k', linewidth=0.8)

a, b = popt

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ax.text(0.05, 0.90, 'y = %.2f*x + %.2f' % (slope, intercept), fontsize=7, transform=plt.gca().transAxes)
ax.text(0.05, 0.80, '$r$ = %.2f' % r_value, fontsize=7, transform=plt.gca().transAxes)
ax.text(0.05, 0.70, 'RMSE = %.2f' % rmse_phenocam, fontsize=7, transform=plt.gca().transAxes)
ax.text(0.05, 0.60, 'p < 0.001', fontsize=7, transform=plt.gca().transAxes)

ax.set_xlabel('Sentinel-2 derived deciduousness(%)', fontsize=7)
ax.set_ylabel('Phenocam derived deciduousness(%)', fontsize=7)

ax.set_xlim(-5, 45)
ax.set_ylim(-5, 45)

ax.tick_params(axis='both', which='major', labelsize=6, length=2)
ax.legend(loc='lower right', fontsize=6, frameon=True, ncol=1)

print('mean r: %.2f' % np.mean(r_list))
print('rmse: %.2f' % np.mean(rmse_list))
