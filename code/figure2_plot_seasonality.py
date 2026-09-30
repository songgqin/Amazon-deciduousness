"""Figure 2 seasonality plotting.

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

# In[] Workflow
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams["axes.unicode_minus"] = True
matplotlib.use("Agg")
matplotlib.rcParams['figure.dpi'] = 300

# In[] Functions
def add_shp(ax, **kwargs):
    proj = ccrs.PlateCarree()

    reader = shpreader.Reader(
        r'data/boundaries/Amazon_ThreeRegions_Clip.shp')

    provinces = reader.geometries()
    ax.add_geometries(provinces, proj, **kwargs)
    reader.close()

def readTif_gdal(fileName):
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    dataset = gdal.Open(fileName)
    if dataset == None:
        print("cannot open file:" + fileName)
        return
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    if len(im_data.shape) == 3:
        if im_data.shape[0] < im_data.shape[2]:
            im_data = np.transpose(im_data, [1, 2, 0])
    return dataset.GetGeoTransform(), dataset.GetProjection(), im_data

def save_tif_int(grouthTif, savePath, Geo_, Projection_, nbands):
    gdal.UseExceptions()

    driver = gdal.GetDriverByName("GTiff")
    driver.Register()

    datatype = gdal.GDT_UInt16
    outputData = driver.Create(savePath, grouthTif.shape[1], grouthTif.shape[0], nbands, datatype,
                               options=["COMPRESS=LZW", 'TILED=YES', 'PREDICTOR=2'])

    outputData.SetGeoTransform(Geo_)
    outputData.SetProjection(Projection_)

    for i in range(nbands):
        outputData.GetRasterBand(i + 1).WriteArray(grouthTif[:, :, i])
        outputData.GetRasterBand(i + 1).SetNoDataValue(65535)

    del outputData

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

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

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

raws_y, colums_x = 786, 650

print(">>> Loading 2019-2021 deciduousness and EVI data to calculate interannual variation...")

rainfall_path = r'data/drivers/inputs/hydroclimate_precipitation_ERA.tif'
_geo, _prj, rainfall = readTif_gdal(rainfall_path)
rainfall = cv2.resize(rainfall.astype(np.float32), (raws_y, colums_x)) * 1000.0

time_lag_path = r'outputs/seasonality/time_lag_map_0521.tif'
_, _, time_lag = readTif_gdal(time_lag_path)

evi_dir = r'data/seasonality'
evi_mean_path = os.path.join(evi_dir, 'BRDF_EVI_3years_mean.tif')
_geo_evi, _, evi_mean_basin = readTif_gdal(evi_mean_path)

rows_evi, cols_evi = evi_mean_basin.shape[0], evi_mean_basin.shape[1]

evi_forest_mask = cv2.resize(forest_mask.astype(np.float32), (cols_evi, rows_evi), interpolation=cv2.INTER_NEAREST).astype(bool)

evi_years_data = []
for year in [2019, 2020, 2021]:
    path = os.path.join(evi_dir, f'BRDF_EVI_{year}.tif')
    _, _, d = readTif_gdal(path)
    evi_years_data.append(d)
evi_stack = np.stack(evi_years_data, axis=0)

dec_dir = r'data/deciduousness'
dec_years_data = []
for year in [2019, 2020, 2021]:
    path = os.path.join(dec_dir, f'Composite_Data_{year}_5km_gf.tif')
    _, _, d = readTif_gdal(path)
    d = d.astype(np.float32)
    d[d > 1000] = np.nan
    dec_years_data.append(d/1000.0)
dec_stack = np.stack(dec_years_data, axis=0)

dec_mean_path = r'data/deciduousness/Composite_Data_5km_gf_3y.tif'
_geo, _prj, dec_mean_basin = readTif_gdal(dec_mean_path)

dec_mean_basin = dec_mean_basin.astype(np.float32)
dec_mean_basin[dec_mean_basin > 1000] = np.nan
dec_mean_basin = dec_mean_basin / 1000.0

print(">>> Loading PhenoCam ground observations...")
phenocam_site = ['BCI', 'K34', 'K67', 'ATTO', 'RJA']
phenocam_dir = r'data/phenocam'

phenocam_dict = {}

for site_name in phenocam_site:
    phenocam_path = os.path.join(phenocam_dir, '{}_Phenocam.csv'.format(site_name))
    if not os.path.exists(phenocam_path):
        continue

    phenocam_file = pd.read_csv(phenocam_path).values

    if site_name == 'BCI':
        phenocam_seasonality = np.nanmean(phenocam_file[:, 1:-1], axis=1)
    elif site_name in ['PEG', 'RJA']:
        phenocam_seasonality = phenocam_file[:, -1]
    elif site_name == 'ATTO':
        if phenocam_file.shape[1] > 2:
            phenocam_seasonality = np.nanmean(phenocam_file[:, 1:-1], axis=1) / 100
        else:
            phenocam_seasonality = 1 - phenocam_file[:, 1]
    elif site_name in ['K34', 'K67']:
        phenocam_seasonality = 1 - phenocam_file[:, 1]

    phenocam_dict[site_name] = phenocam_seasonality

location_path = r'data/phenocam/ATTO_RJA_Location.csv'
loc_df = pd.read_csv(location_path)

ATTO_lat, ATTO_lon = loc_df['Lat'].to_numpy()[0], loc_df['Lon'].to_numpy()[0]
RJA_lat, RJA_lon = loc_df['Lat'].to_numpy()[1], loc_df['Lon'].to_numpy()[1]

ATTO_y, ATTO_x = int((ATTO_lon - _geo[0]) / _geo[1]), int((ATTO_lat - _geo[3]) / _geo[5])
RJA_y, RJA_x = int((RJA_lon - _geo[0]) / _geo[1]), int((RJA_lat - _geo[3]) / _geo[5])

ATTO_y_evi, ATTO_x_evi = int((ATTO_lon - _geo_evi[0]) / _geo_evi[1]), int((ATTO_lat - _geo_evi[3]) / _geo_evi[5])
RJA_y_evi, RJA_x_evi = int((RJA_lon - _geo_evi[0]) / _geo_evi[1]), int((RJA_lat - _geo_evi[3]) / _geo_evi[5])

title_list = ['ATTO', 'RJA']
region_list = [(ATTO_x, ATTO_y), (RJA_x, RJA_y)]
region_list_evi = [(ATTO_x_evi, ATTO_y_evi), (RJA_x_evi, RJA_y_evi)]

month_list = np.arange(12)
month_label_list = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), layout="constrained", dpi=300)
patch_size = 1
tmp_i = 0

for j in range(2):
    id_index = region_list[tmp_i]
    id_evi_index = region_list_evi[tmp_i]

    evi_season_3y = evi_stack[:,
                    id_evi_index[0] - patch_size:id_evi_index[0] + patch_size + 1,
                    id_evi_index[1] - patch_size:id_evi_index[1] + patch_size + 1, :]
    evi_spatial_mean = np.nanmean(evi_season_3y, axis=(1, 2))

    evi_std = np.nanstd(evi_spatial_mean, axis=0)

    evi_patch = evi_mean_basin[id_evi_index[0] - patch_size:id_evi_index[0] + patch_size + 1,
                                id_evi_index[1] - patch_size:id_evi_index[1] + patch_size + 1, :]
    evi_mean = np.nanmean(evi_patch, axis=(0, 1))

    dec_season_3y = dec_stack[:,
                    id_index[0] - patch_size:id_index[0] + patch_size + 1,
                    id_index[1] - patch_size:id_index[1] + patch_size + 1, :]
    dec_spatial_mean = np.nanmean(dec_season_3y, axis=(1, 2))
    dec_std = np.nanstd(dec_spatial_mean, axis=0)

    dec_mean_patch = dec_mean_basin[
                     id_index[0] - patch_size:id_index[0] + patch_size + 1,
                     id_index[1] - patch_size:id_index[1] + patch_size + 1, :]

    dec_mean = np.nanmean(dec_mean_patch, axis=(0, 1))

    rain_season = rainfall[id_index[0] - patch_size:id_index[0] + patch_size + 1,
                  id_index[1] - patch_size:id_index[1] + patch_size + 1]
    rain_mean = np.nanmean(rain_season, axis=(0, 1))

    print('Rainfall:', rain_mean)

    time_lag_patch = time_lag[id_index[0] - patch_size:id_index[0] + patch_size + 1,
                     id_index[1] - patch_size:id_index[1] + patch_size + 1]
    time_lag_mean = np.nanmean(time_lag_patch)
    time_lag_std = np.nanstd(time_lag_patch)
    time_lag_25 = np.nanpercentile(time_lag_patch, 25)
    time_lag_75 = np.nanpercentile(time_lag_patch, 75)
    print('Time lag:', time_lag_mean, time_lag_std, time_lag_25, time_lag_75)

    ax1 = axes[j]

    ax1.errorbar(month_list, dec_mean, yerr=dec_std,
                 fmt='o-', color='k',
                 linewidth=1.5, markersize=4,
                 elinewidth=1.2, capsize=3.5, zorder=3)

    ax1.set_ylabel('Deciduousness (%)', color='k', fontsize=10)

    ax2 = ax1.twinx()

    ax2.errorbar(month_list, evi_mean, yerr=evi_std, fmt='o-', color='g',
                 linewidth=1.5, markersize=4,
                 elinewidth=1.2, capsize=3.5, zorder=3)

    ax2.set_ylabel('EVI', color='g', fontsize=10)
    ax2.yaxis.set_label_coords(1.15, 0.6)

    ax2.set_ylim([0.43, 0.57])
    ax2.set_yticks([0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56])

    ax2.spines['right'].set_color('g')
    ax2.tick_params('y', colors='g')

    ax3 = ax1.twinx()

    ax3.spines['right'].set_position(('outward', 32))
    ax3.yaxis.set_ticks_position('right')
    ax3.yaxis.set_label_position('right')
    ax3.bar(month_list, rain_mean, width=0.9, color='#0C7BDC', label='rainfall', alpha=0.6, zorder=0)

    month_dry_ind = np.where(rain_mean < 100)[0]
    month_dry_list = month_list[month_dry_ind]

    dry_range = np.array([month_dry_list[0] - 0.45, month_dry_list[-1] + 0.45])
    ax3.fill_between(dry_range, 0, 1000, color='grey', alpha=0.21, zorder=0, edgecolor='none')

    ax3.set_ylim([0, 1000])
    ax3.set_yticks([0, 100, 200, 300, 400, ])
    ax3.set_yticklabels([0, 100, 200, 300, 400])
    ax3.spines['right'].set_bounds(0, 400)

    ax3.set_ylabel('Precipitation (mm)', color='#0C7BDC', fontsize=10)
    ax3.yaxis.set_label_coords(1.27, 0.20)

    ax3.spines['right'].set_color('#0C7BDC')

    ax3.tick_params('y', colors='#0C7BDC')

    title = title_list[tmp_i]

    ax1.set_xlabel('Month', fontsize=11)
    ax1.set_xticks(month_list)
    ax1.set_xticklabels(month_label_list, fontsize=10)

    ax1.set_title(title, fontsize=11)

    ax1.set_ylim([-0.06, 0.40])
    ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax1.set_yticklabels([0, 10, 20, 30, 40], fontsize=10)

    ax1.text(0.05, 0.90, r'$\triangle$$\mathit{t}$: %.1f +/- %.1f month' % (time_lag_mean, time_lag_std),
             fontsize=10, transform=ax1.transAxes)

    ax1.text(0.05, 0.80, 'MAP: %0.f mm/year' % np.sum(rain_mean), fontsize=10, color='#0C7BDC',
             transform=ax1.transAxes)

    tmp_i += 1

plt.show()

os.makedirs(r'outputs/figures', exist_ok=True)
save_path = r'outputs/figures/FigMF_2_Seasonality_Dec_EVI.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)

time_lag_path = r'outputs/seasonality/time_lag_map_0521.tif'
_geo, _prj, time_lag = readTif_gdal(time_lag_path)

correlation_path = r'outputs/seasonality/Cor_Dec_EVI_0521.tif'

_, _, correlation_map = readTif_gdal(correlation_path)

region_list = [(ATTO_lat, ATTO_lon), (RJA_lat, RJA_lon)]

cmap = plt.cm.RdBu_r

light_gray = np.array([0.83, 0.83, 0.83, 1.0])
colors = cmap(np.linspace(0, 1, cmap.N))

custom_cmap = mcolors.ListedColormap(colors)

norm = mcolors.PowerNorm(gamma=1.1, vmin=0, vmax=6)

fig = plt.figure(dpi=300)

left, bottom, width, height = 0, 0.1, 0.96, 0.84

proj = ccrs.PlateCarree()
ax = fig.add_subplot([left, bottom, width, height], projection=proj)

ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
ax.add_feature(cfeature.OCEAN, zorder=2)

add_shp(ax, lw=0.5, ec='k', fc='none', zorder=3)

ax.imshow(time_lag, origin='upper',
          extent=[-79.77497863776252, -44.516103736085974, -20.521922834169924, 8.628408135496414],
          transform=ccrs.PlateCarree(), zorder=2, cmap=custom_cmap, norm=norm, )

for tmp_i in np.arange(len(region_list)):
    ax.plot(region_list[tmp_i][1], region_list[tmp_i][0], marker='*', color='k',
            alpha=0.9, markersize=12, markeredgecolor='None', zorder=4)

    ax.text(region_list[tmp_i][1] - 0.6, region_list[tmp_i][0] - 2, title_list[tmp_i].split('_')[0],
            color="k", fontsize=14, )

extents = [-80, -44, -22, 10]
ax.set_extent(extents, crs=proj)

ax.set_xticks(np.arange(-80, -40, 10), crs=proj)
ax.set_yticks(np.arange(-20, 10 + 10, 10), crs=proj)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.tick_params(axis='both', labelsize=12, direction='out', colors='k', length=3, width=0.9, which='major',
               left=True, right=False, top=True, bottom=False,
               labelleft=True, labelright=False, labeltop=True, labelbottom=False, zorder=10)

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax, orientation='horizontal',
                    fraction=0.04, pad=0.03)

ax.text(-78.4, -24, 'Low', clip_on=True, transform=ccrs.PlateCarree())
ax.text(-47.5, -24, 'High', clip_on=True, transform=ccrs.PlateCarree())

cbar.set_label(r'$\triangle$$\mathit{t}$ (month)', fontsize=12)

save_path = r'outputs/figures/Time_Lag_Map_phenocam_0521.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)

cmap = plt.cm.BrBG

light_gray = np.array([0.83, 0.83, 0.83, 1.0])
colors = cmap(np.linspace(0, 1, cmap.N))

custom_cmap = mcolors.ListedColormap(colors)

norm = mcolors.PowerNorm(gamma=1, vmin=-1, vmax=1)

fig = plt.figure(dpi=300)

left, bottom, width, height = 0, 0.1, 0.96, 0.84

proj = ccrs.PlateCarree()
ax = fig.add_subplot([left, bottom, width, height], projection=proj)

ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
ax.add_feature(cfeature.OCEAN, zorder=2)

add_shp(ax, lw=0.5, ec='k', fc='none', zorder=3)

ax.imshow(correlation_map, origin='upper',
          extent=[-79.77497863776252, -44.516103736085974, -20.521922834169924, 8.628408135496414],
          transform=ccrs.PlateCarree(), zorder=2, cmap=custom_cmap, norm=norm)

for tmp_i in np.arange(len(region_list)):
    ax.plot(region_list[tmp_i][1], region_list[tmp_i][0], marker='*', color='k',
            alpha=0.9, markersize=12, markeredgecolor='None', zorder=4)

    ax.text(region_list[tmp_i][1] - 0.6, region_list[tmp_i][0] - 2, title_list[tmp_i].split('_')[0],
            color="k", fontsize=14, )

extents = [-80, -44, -22, 10]
ax.set_extent(extents, crs=proj)

ax.set_xticks(np.arange(-80, -40, 10), crs=proj)
ax.set_yticks(np.arange(-20, 10 + 10, 10), crs=proj)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.tick_params(axis='both', labelsize=12, direction='out', colors='k', length=3, width=0.9, which='major',
               left=True, right=False, top=True, bottom=False,
               labelleft=True, labelright=False, labeltop=True, labelbottom=False, zorder=10)

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax, orientation='horizontal',
                    fraction=0.04, pad=0.03)

cbar.set_label('Correlation coefficient', fontsize=12)

save_path = r'outputs/figures/EVI_Dec_Cor_phenocam_0521.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)
