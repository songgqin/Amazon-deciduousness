import os
import copy
import cv2
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
# from patchify import patchify
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
# import cartopy.crs as ccrs
# import gdal
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import scipy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# 导入底图包
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from skimage.measure import find_contours
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams["axes.unicode_minus"] = True  # 显示负号
matplotlib.use('Qt5Agg')
matplotlib.rcParams['figure.dpi'] = 300


# In[]
def add_shp(ax, **kwargs):
    '''
    在地图上画出中国省界的shapefile.

    Parameters
    ----------
    ax : GeoAxes
        目标地图.

    **kwargs
        绘制shape时用到的参数.例如linewidth,edgecolor和facecolor等.
    '''
    proj = ccrs.PlateCarree()
    # reader = shpreader.Reader(r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\AmazonBasin_Shapefile\amazon_sensulatissimo_gmm_v1.shp')
    reader = shpreader.Reader(
        r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\ThreeRegions_Boundary\clip\Amazon_ThreeRegions_Clip.shp')

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
    # transpose
    if len(im_data.shape) == 3:
        if im_data.shape[0] < im_data.shape[2]:
            im_data = np.transpose(im_data, [1, 2, 0])
    return dataset.GetGeoTransform(), dataset.GetProjection(), im_data


def save_tif_int(grouthTif, savePath, Geo_, Projection_, nbands):
    gdal.UseExceptions()

    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    # datatype = gdal.GDT_Float32
    datatype = gdal.GDT_UInt16
    outputData = driver.Create(savePath, grouthTif.shape[1], grouthTif.shape[0], nbands, datatype,
                               options=["COMPRESS=LZW", 'TILED=YES', 'PREDICTOR=2'])

    # if (outputData != None):
    outputData.SetGeoTransform(Geo_)  # 写入仿射变换参数
    outputData.SetProjection(Projection_)  # 写入投影
    # print('done')

    # Write in the DataValue
    # if nbands == 1:
    #     outputData.GetRasterBand(1).WriteArray(grouthTif)
    #     outputData.GetRasterBand(1).SetNoDataValue(65535)
    #     # outputData.GetRasterBand(1).SetNoDataValue(np.nan)
    # else:
    for i in range(nbands):
        outputData.GetRasterBand(i + 1).WriteArray(grouthTif[:, :, i])
        outputData.GetRasterBand(i + 1).SetNoDataValue(65535)
        # outputData.GetRasterBand(i + 1).SetNoDataValue(np.nan)

    del outputData


def save_tif(grouthTif, savePath, Geo_, Projection_, nbands):
    gdal.UseExceptions()
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    datatype = gdal.GDT_Float32
    # datatype = gdal.GDT_UInt16
    outputData = driver.Create(savePath, grouthTif.shape[1], grouthTif.shape[0], nbands, datatype)

    # if (outputData != None):
    outputData.SetGeoTransform(Geo_)  # 写入仿射变换参数
    outputData.SetProjection(Projection_)  # 写入投影
    # print('done')

    # Write in the DataValue
    if nbands == 1:
        outputData.GetRasterBand(1).WriteArray(grouthTif)
        outputData.GetRasterBand(1).SetNoDataValue(np.nan)
        # outputData.GetRasterBand(1).SetNoDataValue(9999)
    else:
        for i in range(nbands):
            outputData.GetRasterBand(i + 1).WriteArray(grouthTif[:, :, i])
            outputData.GetRasterBand(1).SetNoDataValue(np.nan)
        # outputData.GetRasterBand(i + 1).SetNoDataValue(9999)
    del outputData


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


# In[]
raws_y, columns_x = 786, 650
cls_modis_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig1\Data\Forest_Mask\MCD12Q1_Amazon.tif'
# cls_modis_path = r'X:\Song_Amazon_Mapping\classification_map.tif'
_, _, cls_md = readTif_gdal(cls_modis_path)
cls_md_forest = copy.deepcopy(cls_md)
cls_md_forest = cls_md_forest.astype(np.float32)
cls_md_forest[cls_md_forest != 2] = np.nan
cls_modis_coarse = cv2.resize(cls_md_forest, (raws_y, columns_x), interpolation=cv2.INTER_NEAREST)

forest_mask = ~np.isnan(cls_modis_coarse)  # the mask of forest pixels
# forest_mask = copy.deepcopy(cls_modis_coarse)
# valid evergreen forest pixels
ind_where = np.where(forest_mask == 1)  # find the indices of evergreen forest pixels
print('evergreen forest pixels:', len(ind_where[0]))

# In[] Visualize the seasonal pattern of EVI, rainfall, and deciduousness
raws_y, colums_x = 786, 650

seasonality_example_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Figures\Example_Mask.tif'
# r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Seasonal_Example3.tif'
_geo, _prj, season_mask = readTif_gdal(seasonality_example_path)
season_mask = season_mask.astype(np.float32)
season_mask[season_mask == 15] = np.nan
season_mask = cv2.resize(season_mask, (raws_y, colums_x))

EVI_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Data\seasonality\BRDF_EVI.tif'  # MAIACI_EVI.tif
# EVI_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Data\seasonality\MAIACI_EVI.tif'
# EVI_path = r'E:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Data\seasonality\MCD43A4_EVI.tif'
_geo_evi, _prj_evi, EVI_amazon = readTif_gdal(EVI_path)

EVI_amazon = cv2.resize(EVI_amazon, (raws_y, colums_x))
EVI_amazon = EVI_amazon#/10000.0

# rainfall_path = r'E:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Data\seasonality\MCD43A4_EVI.tif'
# _geo, _prj, rainfall = readTif_gdal(rainfall_path)
# rainfall = cv2.resize(rainfall, (raws_y, colums_x))

# rainfall_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Data\seasonality\precipiation.tif'

rainfall_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Data\seasonality\hydroclimate_precipitation_ERA.tif'
_geo, _prj, rainfall = readTif_gdal(rainfall_path)
rainfall = rainfall.astype(np.float32)
rainfall = cv2.resize(rainfall, (raws_y, colums_x))

rainfall = rainfall * 1000.0
# dec_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Multiyear_mosaic\250m_gapfill\v2_threshold_0.45_overestimate\95qc_DSO_1111\Mean_Filled_95qc\Composite_Data_5km_gf_3y.tif'
dec_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Data\seasonality\Deciduousness\Composite_Data_5km_gf_3y.tif'
_geo, _prj, dec_month = readTif_gdal(dec_path)

dec_month = dec_month.astype(np.float32)

dec_month[dec_month > 1000] = np.nan

dec_month[forest_mask == False] = np.nan

# time_lag_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Python-Based Figures\Tif_Data\time_lag_map.tif'

time_lag_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Python-Based Figures\Tif_Data\time_lag_map_0521.tif'

_, _, time_lag = readTif_gdal(time_lag_path)

# In[]
# region_list = [4, 1]
location_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Phenocam_GCC\ATTO_RJA_Location.csv'
loc_df = pd.read_csv(location_path)

ATTO_lat = loc_df['Lat'].to_numpy()[0]
ATTO_lon = loc_df['Lon'].to_numpy()[0]

RJA_lat = loc_df['Lat'].to_numpy()[1]
RJA_lon = loc_df['Lon'].to_numpy()[1]

# extract the EVI value at the phenocam site based on the lat and lon
ATTO_y, ATTO_x = int((ATTO_lon - _geo[0]) / _geo[1]), int((ATTO_lat - _geo[3]) / _geo[5])
RJA_y, RJA_x = int((RJA_lon - _geo[0]) / _geo[1]), int((RJA_lat - _geo[3]) / _geo[5])

ATTO_y_evi, ATTO_x_evi = int((ATTO_lon - _geo_evi[0]) / _geo_evi[1]), int((ATTO_lat - _geo_evi[3]) / _geo_evi[5])
RJA_y_evi, RJA_x_evi = int((RJA_lon - _geo_evi[0]) / _geo_evi[1]), int((RJA_lat - _geo_evi[3]) / _geo_evi[5])

# ATTO_EVI = np.nanmean(EVI_amazon[ATTO_x - patch_size:ATTO_x + patch_size + 1, ATTO_y - patch_size:ATTO_y + patch_size + 1], axis=(0, 1))
# RJA_EVI = np.nanmean(EVI_amazon[RJA_x - patch_size:RJA_x + patch_size + 1, RJA_y - patch_size:RJA_y + patch_size + 1], axis=(0, 1))
#

# region_list = [1, 4]

# title_list = ['High Asynchrony', 'Low Asynchrony']

title_list = ['ATTO', 'RJA']

region_list = [(ATTO_x, ATTO_y), (RJA_x, RJA_y)]

region_list_evi = [(ATTO_x_evi, ATTO_y_evi), (RJA_x_evi, RJA_y_evi)]

month_list = np.arange(12)
month_label_list = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
# fig, axes = plt.subplots(2, 2, figsize=(9.6,4.8), dpi=300)  # ,sharex=True, sharey=True,)

# In[]
fig, axes = plt.subplots(1, 2, figsize=(9, 3))  # ,sharex=True, sharey=True,)
tmp_i = 0

patch_size = 1

for i in range(1):
    for j in range(2):
        # id = region_list[tmp_i]
        id_index = region_list[tmp_i]
        # evi_season = EVI_amazon[id_index[0] - patch_size:id_index[0] + patch_size + 1,
        #              id_index[1] - patch_size:id_index[1] + patch_size + 1]

        id_evi_index = region_list_evi[tmp_i]
        evi_season = EVI_amazon[id_evi_index[0] - patch_size:id_evi_index[0] + patch_size + 1,
                     id_evi_index[1] - patch_size:id_evi_index[1] + patch_size + 1]

        rain_season = rainfall[id_index[0] - patch_size:id_index[0] + patch_size + 1,
                      id_index[1] - patch_size:id_index[1] + patch_size + 1]
        dec_season = dec_month[id_index[0] - patch_size:id_index[0] + patch_size + 1,
                     id_index[1] - patch_size:id_index[1] + patch_size + 1]

        time_lag_mean = np.nanmean(time_lag[id_index[0] - patch_size:id_index[0] + patch_size + 1,
                                   id_index[1] - patch_size:id_index[1] + patch_size + 1])
        time_lag_std = np.nanstd(time_lag[id_index[0] - patch_size:id_index[0] + patch_size + 1,
                                 id_index[1] - patch_size:id_index[1] + patch_size + 1])

        time_lag_25 = np.nanpercentile(time_lag[id_index[0] - patch_size:id_index[0] + patch_size + 1,
                                       id_index[1] - patch_size:id_index[1] + patch_size + 1], 25)
        time_lag_75 = np.nanpercentile(time_lag[id_index[0] - patch_size:id_index[0] + patch_size + 1,
                                       id_index[1] - patch_size:id_index[1] + patch_size + 1], 75)
        # rain_season = rainfall[season_mask == id]
        # dec_season = dec_month[season_mask == id]
        # time_lag_mean = np.nanmean(time_lag[season_mask == id])
        # time_lag_std = np.nanstd(time_lag[season_mask == id])
        # time_lag_25 = np.nanpercentile(time_lag[season_mask == id], 25)
        # time_lag_75 = np.nanpercentile(time_lag[season_mask == id], 75)
        print('Time lag:', time_lag_mean, time_lag_std, time_lag_25, time_lag_75)

        evi_mean = np.nanmean(evi_season, axis=(0, 1))
        evi_std = np.nanstd(evi_season, axis=(0, 1))

        evi_min = evi_mean - evi_std
        evi_max = evi_mean + evi_std

        rain_mean = np.nanmean(rain_season, axis=(0, 1))
        rain_std = np.nanstd(rain_season, axis=(0, 1))
        rain_min = rain_mean - rain_std
        rain_max = rain_mean + rain_std

        dec_mean = np.nanmean(dec_season, axis=(0, 1))
        dec_std = np.nanstd(dec_season, axis=(0, 1))
        # dec_mean, dec_min, dec_max = mean_confidence_interval(dec_season, confidence=0.95)
        # 95% confidence interval
        # dec_95 = 1.96 * dec_std / np.sqrt(np.sum(~np.isnan(dec_season), axis=0))
        dec_min = dec_mean - dec_std
        dec_max = dec_mean + dec_std

        # fig, ax1 = plt.subplots(layout="constrained", dpi=300)  # ,sharex=True, sharey=True,)

        ax1 = axes[j]  # deciduousness
        # fig, col = plt.subplots()  # plt.figure()
        # ax1.plot(month_list, dec_mean / 1000.0, color='k', label='deciduousness fraction', zorder=1)
        ax1.errorbar(month_list, dec_mean / 1000.0, yerr=dec_std / 1000.0, fmt='o-', color='k', elinewidth=1, zorder=1)
        # ax1.fill_between(month_list, dec_min / 1000.0, dec_max / 1000.0, color='k', alpha=0.2, zorder=1)
        ax1.set_ylabel('Deciduousness(%)', color='k', fontsize=10)  # fraction
        # del dec_mean, dec_std, dec_min, dec_max

        # col.plot(month_list, evi_mean / 10000.0, color='g', label='EVI')
        # col.fill_between(month_list, evi_min / 10000.0, evi_max / 10000.0, color='g', alpha=0.2)

        ax2 = ax1.twinx()
        # evi_mean = savgol_filter(evi_mean, 3, 1)  # savgol_filter(evi_mean, 3, 1)
        # ax2.plot(month_list, evi_mean, color='g', label='EVI', zorder=1)
        ax2.errorbar(month_list, evi_mean, yerr=evi_std, fmt='o-', color='g', elinewidth=1, zorder=1)

        # ax2.fill_between(month_list, evi_min, evi_max, color='g', alpha=0.2, zorder=1)
        # # 调整ax2的位置，为了不与ax1重叠
        # ax2.spines['right'].set_position(('outward', 0))
        # ax2.yaxis.set_ticks_position('right')
        # ax2.yaxis.set_label_position('right')
        # ax2.plot(month_list, evi_mean, color='g', label='EVI', zorder=1)
        # ax2.fill_between(month_list, evi_min, evi_max, color='g', alpha=0.2, zorder=1)
        #
        ax2.set_ylabel('EVI', color='g', fontsize=10)
        ax2.yaxis.set_label_coords(1.15, 0.6)

        ax2.set_ylim([0.43, 0.57])
        ax2.set_yticks([0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56])
        ax2.spines['right'].set_color('g')
        ax2.tick_params('y', colors='g')

        ax3 = ax1.twinx()
        # # 调整ax2的位置，为了不与ax1重叠
        ax3.spines['right'].set_position(('outward', 32))
        ax3.yaxis.set_ticks_position('right')
        ax3.yaxis.set_label_position('right')
        ax3.bar(month_list, rain_mean, width=0.9, color='#0C7BDC', label='rainfall', alpha=0.6, zorder=0)

        # dry season < 100mm
        month_dry_ind = np.where(rain_mean < 100)[0]
        month_dry_list = month_list[month_dry_ind]

        dry_range = np.array([month_dry_list[0] - 0.45, month_dry_list[-1] + 0.45])
        ax3.fill_between(dry_range, 0, 1000, color='grey', alpha=0.21, zorder=0, edgecolor='none')  # '#fdae61'

        ax3.set_ylim([0, 1000])
        ax3.set_yticks([0, 100, 200, 300, 400, ])
        ax3.set_yticklabels([0, 100, 200, 300, 400])
        ax3.spines['right'].set_bounds(0, 400)

        # ax3.set_ylabel('rainfall$ \ mm$', color='b')

        ax3.set_ylabel('Precipitation(mm)', color='#0C7BDC', fontsize=10)
        ax3.yaxis.set_label_coords(1.27, 0.20)  # 0.15

        ax3.spines['right'].set_color('#0C7BDC')

        ax3.tick_params('y', colors='#0C7BDC')

        title = title_list[tmp_i]

        ax1.set_xlabel('Month', fontsize=11)
        ax1.set_xticks(month_list)
        ax1.set_xticklabels(month_label_list, fontsize=10)

        ax1.set_title(title, fontsize=11)
        # col.set_xlim([0, 11])

        ax1.set_ylim([-0.06, 0.40])  # D%
        ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        ax1.set_yticklabels([0, 10, 20, 30, 40], fontsize=10)

        ax1.text(0.05, 0.90, r'$\triangle$$\mathit{t}$: %.1f ± %.1f(month)' % (time_lag_mean, time_lag_std),
                 fontsize=10, transform=ax1.transAxes)
        # ax1.text(-0.35, 0.33, r'$\triangle$$\mathit{t}$: %.0f(month)' % (time_lag_mean), fontsize=10)
        ax1.text(0.05, 0.80, 'MAP: %0.f(mm/ year)' % np.sum(rain_mean), fontsize=10, color='#0C7BDC',
                 transform=ax1.transAxes)

        # if tmp_i == 1: # RJA
        #     ax1.set_ylim([-0.06, 0.40]) # D%
        #     ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        #     ax1.set_yticklabels([0, 10, 20, 30, 40], fontsize=10)
        #     ax1.text(-0.35, 0.345, r'$\triangle$$\mathit{t}$: %.1f ± %.1f(month)' % (time_lag_mean, time_lag_std), fontsize=10,transform=ax1.transAxes  )
        #     # ax1.text(-0.35, 0.33, r'$\triangle$$\mathit{t}$: %.0f(month)' % (time_lag_mean), fontsize=10)
        #     ax1.text(-0.35, 0.310, 'MAP: %0.f(mm/year)' % np.sum(rain_mean), fontsize=10, color='#0C7BDC',transform=ax1.transAxes)
        #     # ax1.text(-0.35, 0.26, 'DSL: %0.f (month)' % np.sum(rain_mean < 100), fontsize=10, color='#fdae61')
        #
        #     # ax2.set_ylim([0.44, 0.56]) # EVI
        #     # ax2.set_yticks([0.44, 0.48, 0.52, 0.56])
        #
        # else: # ATTO
        #     ax1.set_ylim([-0.06, 0.40]) # D%
        #     ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        #     ax1.set_yticklabels([0, 10, 20, 30, 40], fontsize=10)
        #
        #     ax1.text(-0.35, 0.187, r'$\triangle$$\mathit{t}$: %.1f ± %.1f(month)' % (time_lag_mean, time_lag_std), fontsize=10)
        #     ax1.text(-0.35, 0.167, 'MAP: %0.f(mm/year)' % np.sum(rain_mean), fontsize=10, color='#0C7BDC')
        #     # ax1.text(-0.35, 0.112, 'DSL: %0.f (month)' % np.sum(rain_mean < 100), fontsize=10, color='#fdae61')
        #
        #     ax2.set_ylim([0.43, 0.51]) # EVI
        #     # ax2.set_yticks([0.44, 0.46, 0.48, 0.50])

        # plt.tight_layout()
        tmp_i += 1

plt.tight_layout()

# save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Figures\Example_Seasonality_ATTO_RJA_0521.png'
# plt.savefig(save_path, bbox_inches='tight', dpi=300)
# In[] show the time lag and correlation between EVI and deciduousness map

# read the data
# time_lag_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Python-Based Figures\Tif_Data\time_lag_map.tif'
time_lag_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Python-Based Figures\Tif_Data\time_lag_map_0521.tif'
# time_lag_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Python-Based Figures\Tif_Data\kNDVI_time_lag_map.tif'
# time_lag_path = r'Deciduousness_Ground_Validation\Time_lag_Comparison\Peak_to_Peak_Dec_EVI_Time_Lag.tif'
_geo, _prj, time_lag = readTif_gdal(time_lag_path)

# correlation_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Python-Based Figures\Tif_Data\Cor_Dec0803_EVI_BRDF.tif'
correlation_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Python-Based Figures\Tif_Data\Cor_Dec_EVI_0521.tif'
_, _, correlation_map = readTif_gdal(correlation_path)

# generate the boundary of the example region
# region_list = [4, 1]

# region_list = [(ATTO_x, ATTO_y), (RJA_x, RJA_y)]

region_list = [(ATTO_lat, ATTO_lon), (RJA_lat, RJA_lon)]
# plt.colorbar()

# In[] Visualize the time lag map
cmap = plt.cm.RdBu_r
# norm = mcolors.Normalize(vmin=-1, vmax=500)

# light_gray = np.array([0.83, 0.83, 0.83, 1.0])
# colors = cmap(np.linspace(0, 1, cmap.N))
# colors[0] = np.array([0.83, 0.83, 0.83, 1.0])  # Set the first color to gray (for the value -2)
# custom_cmap = mcolors.ListedColormap(colors)
# norm = mcolors.PowerNorm(gamma=1.1, vmin=0, vmax=6)
n_categories = 7
colors = cmap(np.linspace(0.1, 0.8, n_categories))  # 均匀采样 7 个颜色

# 创建 ListedColormap
custom_cmap = mcolors.ListedColormap(colors)

# 定义类别边界：每个整数一个类别 [0,1), [1,2), ..., [6,7)
bounds = np.arange(0, n_categories + 1)  # [0,1,2,3,4,5,6,7]
norm = mcolors.BoundaryNorm(bounds, ncolors=n_categories, clip=False)

# x_0,y_0 = np.where(time_lag == 6)
# plt.imshow(time_lag, cmap=custom_cmap, norm=norm)
# plt.scatter(y_0,x_0,marker='.',s=1)


fig = plt.figure(dpi=300)

##### main axis
# left, bottom, width, height = 0, 0.1, 0.96, 0.84  # the position of left lower corner, width and height of the figure
left, bottom, width, height = 0, 0.1, 0.96, 0.80  # the position of left lower corner, width and height of the figure

proj = ccrs.PlateCarree()
ax = fig.add_subplot([left, bottom, width, height], projection=proj)

# set land and ocean features
ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
ax.add_feature(cfeature.OCEAN, zorder=2)
# add amazon basin boundary
add_shp(ax, lw=0.5, ec='k', fc='none', zorder=3)

ax.imshow(time_lag, origin='upper',
          extent=[-79.77497863776252, -44.516103736085974, -20.521922834169924, 8.628408135496414],
          transform=ccrs.PlateCarree(), zorder=2, cmap=custom_cmap, norm=norm, )  # norm=norm,

# 设置显示经纬度范围, 注意指定crs关键字,否则范围不一定完全准确
# for tmp_i in np.arange(len(region_list)):
#     ax.plot(region_list[tmp_i][1], region_list[tmp_i][0], marker='*', color='k',  #
#             alpha=0.9, markersize=12, markeredgecolor='None', zorder=4)  # markeredgecolor='None'
#
#     ax.text(region_list[tmp_i][1] - 0.6, region_list[tmp_i][0] - 2, title_list[tmp_i].split('_')[0],
#             color="k", fontsize=10, )  # 半透明黑底白字

# # add the boundary of the example region
# ax.plot(boundary_1_geo[:, 0], boundary_1_geo[:, 1], color='k', zorder=4, linewidth=1.5)  # high asynchrony
# ax.plot(boundary_2_geo[:, 0], boundary_2_geo[:, 1], color='darkred', zorder=4, linewidth=1.5)  # low asynchrony

extents = [-80, -44, -22, 10]
ax.set_extent(extents, crs=proj)

## 设置经纬度major & minor刻度
ax.set_xticks(np.arange(-80, -40, 10), crs=proj)
ax.set_yticks(np.arange(-20, 10 + 10, 10), crs=proj)
## 添加设置网格线
# ax.grid(color=[0.94, 0.94, 0.94], linestyle='--', zorder=1)
## 利用Formatter格式化刻度标签
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

# ax.tick_params(axis='both', labelsize=12, direction='out', colors='k', length=3, width=0.9, which='major',
#                left=False, right=True, top=True, bottom=False,
#                labelleft=False, labelright=True, labeltop=True, labelbottom=False, zorder=10)

ax.tick_params(axis='both', labelsize=12, direction='out', colors='k', length=3, width=0.9, which='major',
               left=True, right=False, top=True, bottom=False,
               labelleft=True, labelright=False, labeltop=True, labelbottom=False, zorder=10)

# draw a white rectangle around the subplots
left, bottom, width, height = 0.68, 0.145, 0.30, 0.30
ax0 = fig.add_axes([left, bottom, width, height])
plt.Rectangle((left, bottom), width, height, transform=ax0.transAxes, facecolor='white', zorder=3)
ax0.tick_params(axis='both', which='both', colors='white', left=False, right=False, top=False, bottom=False,
                labelleft=False, labelright=False, labeltop=False, labelbottom=False, )

ax0.spines['bottom'].set_color('white')
ax0.spines['top'].set_color('white')
ax0.spines['right'].set_color('white')
ax0.spines['left'].set_color('white')

# plt.axis('off')

# ax0.set_facecolor('white')
# ax0.axis('off')

left, bottom, width, height = 0.70, 0.23, 0.20, 0.20
ax1 = fig.add_axes([left, bottom, width, height])

ax1.set_facecolor('white')

# 使用 sns.histplot 绘制概率直方图
sns.histplot(
    data=time_lag[~np.isnan(time_lag)],
    stat="probability",
    bins=np.arange(-0.5, 7.5, 1),  # 关键：定义 bin 边界，使每个整数为中心
    color='gray',
    alpha=0.6,
    edgecolor='k',
    linewidth=0.25,
    ax=ax1,
    zorder=3
)

# 设置 y 轴范围
ax1.set_ylim(0, 0.30)

# y 轴标签在右侧
ax1.yaxis.set_ticks_position('right')
ax1.yaxis.set_label_position('right')

# 设置 x 轴范围和刻度
ax1.set_xlim(-0.6, 6.6)
ax1.set_xticks(np.arange(0, 7))  # 刻度在整数位置（即每个柱子中心）
ax1.set_xticklabels(np.arange(0, 7), fontsize=10, ha='center')  # 水平居中

# 标签
ax1.set_ylabel('Density', fontsize=10)
ax1.set_xlabel(r'$\triangle$$\mathit{t}$ (month)', fontsize=10, labelpad=0.5)

cbar = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap),
    ax=ax,
    orientation='horizontal',
    fraction=0.04,
    pad=0.03,
    boundaries=bounds,  # 必须指定边界
    ticks=np.arange(0, 7) + 0.5,  # 标签居中
    drawedges=True,  # 显示分隔线（关键！）
    spacing='uniform'  # 确保每段等宽
)

# 设置标签
cbar.set_label(r'$\triangle$$\mathit{t}$ (month)', fontsize=12)
cbar.ax.set_xticklabels(['0', '1', '2', '3', '4', '5', '6'])  # 显示整数标签
cbar.ax.tick_params(length=0)  # 可选：隐藏刻度线，更简洁

# Low time lag for the low asynchrony degree, and high time lag for the high asynchrony degree
ax.text(-78.4, -24, 'Low', clip_on=True, transform=ccrs.PlateCarree())
ax.text(-47.5, -24, 'High', clip_on=True, transform=ccrs.PlateCarree())

plt.show()

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Figures\Time_Lag_Map_phenocam_1003.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)

# In[] Visualize the correlation map

cmap = plt.cm.BrBG
# norm = mcolors.Normalize(vmin=-1, vmax=500)

light_gray = np.array([0.83, 0.83, 0.83, 1.0])
colors = cmap(np.linspace(0, 1, cmap.N))
# colors[0] = np.array([0.83, 0.83, 0.83, 1.0])  # Set the first color to gray (for the value -2)
custom_cmap = mcolors.ListedColormap(colors)

norm = mcolors.PowerNorm(gamma=1, vmin=-1, vmax=1)

# norm = mcolors.Normalize( vmin=-1, vmax=1)

fig = plt.figure(dpi=300)

##### main axis
left, bottom, width, height = 0, 0.1, 0.96, 0.84  # the position of left lower corner, width and height of the figure

proj = ccrs.PlateCarree()
ax = fig.add_subplot([left, bottom, width, height], projection=proj)

# set land and ocean features
ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
ax.add_feature(cfeature.OCEAN, zorder=2)
# add amazon basin boundary
add_shp(ax, lw=0.5, ec='k', fc='none', zorder=3)

ax.imshow(correlation_map, origin='upper',
          extent=[-79.77497863776252, -44.516103736085974, -20.521922834169924, 8.628408135496414],
          transform=ccrs.PlateCarree(), zorder=2, cmap=custom_cmap, norm=norm)

# 设置显示经纬度范围, 注意指定crs关键字,否则范围不一定完全准确

# add the boundary of the example region
# ax.plot(boundary_1_geo[:, 0], boundary_1_geo[:, 1], color='k', zorder=4, linewidth=1.5)  # high asynchrony
# ax.plot(boundary_2_geo[:, 0], boundary_2_geo[:, 1], color='darkred', zorder=4, linewidth=1.5)  # low asynchrony

for tmp_i in np.arange(len(region_list)):
    ax.plot(region_list[tmp_i][1], region_list[tmp_i][0], marker='*', color='k',  #
            alpha=0.9, markersize=12, markeredgecolor='None', zorder=4)  # markeredgecolor='None'

    ax.text(region_list[tmp_i][1] - 0.6, region_list[tmp_i][0] - 2, title_list[tmp_i].split('_')[0],
            color="k", fontsize=14, )  # 半透明黑底白字

extents = [-80, -44, -22, 10]
ax.set_extent(extents, crs=proj)

## 设置经纬度major & minor刻度
ax.set_xticks(np.arange(-80, -40, 10), crs=proj)
ax.set_yticks(np.arange(-20, 10 + 10, 10), crs=proj)
## 添加设置网格线
# ax.grid(color=[0.94, 0.94, 0.94], linestyle='--', zorder=1)
## 利用Formatter格式化刻度标签
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

# ax.tick_params(axis='both', labelsize=12, direction='out', colors='k', length=3, width=0.9, which='major',
#                left=False, right=True, top=True, bottom=False,
#                labelleft=False, labelright=True, labeltop=True, labelbottom=False, zorder=10)

ax.tick_params(axis='both', labelsize=12, direction='out', colors='k', length=3, width=0.9, which='major',
               left=True, right=False, top=True, bottom=False,
               labelleft=True, labelright=False, labeltop=True, labelbottom=False, zorder=10)

# legend_elements = [Patch(facecolor=light_gray, edgecolor='k', label='Non-Forest')]
# ax.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=False, prop={'size': 12})

# draw a white rectangle around the subplots
left, bottom, width, height = 0.68, 0.145, 0.30, 0.30
ax0 = fig.add_axes([left, bottom, width, height])
plt.Rectangle((left, bottom), width, height, transform=ax0.transAxes, facecolor='white', zorder=3)
ax0.tick_params(axis='both', which='both', colors='white', left=False, right=False, top=False, bottom=False,
                labelleft=False, labelright=False, labeltop=False, labelbottom=False, )

ax0.spines['bottom'].set_color('white')
ax0.spines['top'].set_color('white')
ax0.spines['right'].set_color('white')
ax0.spines['left'].set_color('white')

# plt.axis('off')

# ax0.set_facecolor('white')
# ax0.axis('off')

left, bottom, width, height = 0.70, 0.23, 0.20, 0.20
ax1 = fig.add_axes([left, bottom, width, height])

ax1.set_facecolor('white')

# 使用 sns.histplot 绘制概率直方图
sns.histplot(
    data=correlation_map[~np.isnan(correlation_map)],
    stat="probability",
    bins=50, color='gray', alpha=0.6, edgecolor='k', linewidth=0.25, kde=False,
    ax=ax1,
    zorder=3
)

# 设置 y 轴范围
ax1.set_ylim(0, 0.04)

# y 轴标签在右侧
ax1.yaxis.set_ticks_position('right')
ax1.yaxis.set_label_position('right')

# 设置 x 轴范围和刻度
ax1.set_xlim(-1, 1)
# ax1.set_xticks(np.arange(0, 7))  # 刻度在整数位置（即每个柱子中心）
# ax1.set_xticklabels(np.arange(0, 7), fontsize=10, ha='center')  # 水平居中

# 标签
ax1.set_ylabel('Density', fontsize=10)
ax1.set_xlabel(r'Correlation coefficient', fontsize=10, labelpad=0.5)

# show the colorbar of the deciduousness amplitude
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax, orientation='horizontal',
                    fraction=0.04, pad=0.03)

cbar.set_label('Correlation coefficient', fontsize=12)

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Figures\EVI_Dec_Cor_phenocam_0817.png'
# plt.savefig(save_path, bbox_inches='tight', dpi=300)

# In[] proportion area of different time lag

unique, counts = np.unique(time_lag[~np.isnan(time_lag)], return_counts=True)
total_count = np.sum(counts)
proportions = counts / total_count
for u, c, p in zip(unique, counts, proportions):
    print(f'Time lag: {u}, Count: {c}, Proportion: {p:.4f}')

# more than 3 months
count_more_than_3 = np.sum(counts[unique >= 3])
proportion_more_than_3 = count_more_than_3 / total_count
print(f'Count of time lag > 3 months: {count_more_than_3}, Proportion: {proportion_more_than_3:.4f}')

time_lag_res =  time_lag[~np.isnan(time_lag)]

print('Percentage of pixels with time lag > 3 months: {:.2f}%'.format(np.sum(time_lag_res >= 3) / len(time_lag_res) * 100))

# In[] time lag plus p-value visualization
# p-value path
p_value_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Python-Based Figures\Tif_Data\p_value_map_time_lag_cor.tif'
_geo, _prj, p_value_map = readTif_gdal(p_value_path)
ds = gdal.Open(p_value_path)

# Extract affine transform parameters
x_origin, x_res, x_rot, y_origin, y_rot, y_res = _geo
width, height = ds.RasterXSize, ds.RasterYSize

# Generate coordinate arrays
x_indices = np.arange(width)
y_indices = np.arange(height)
lon = x_origin + x_indices * x_res
lat = y_origin + y_indices * y_res

# In[]

cmap = plt.cm.RdBu_r
# norm = mcolors.Normalize(vmin=-1, vmax=500)

# light_gray = np.array([0.83, 0.83, 0.83, 1.0])
# colors = cmap(np.linspace(0, 1, cmap.N))
# colors[0] = np.array([0.83, 0.83, 0.83, 1.0])  # Set the first color to gray (for the value -2)
# custom_cmap = mcolors.ListedColormap(colors)
# norm = mcolors.PowerNorm(gamma=1.1, vmin=0, vmax=6)
n_categories = 7
colors = cmap(np.linspace(0.1, 0.8, n_categories))  # 均匀采样 7 个颜色

# 创建 ListedColormap
custom_cmap = mcolors.ListedColormap(colors)

# 定义类别边界：每个整数一个类别 [0,1), [1,2), ..., [6,7)
bounds = np.arange(0, n_categories + 1)  # [0,1,2,3,4,5,6,7]
norm = mcolors.BoundaryNorm(bounds, ncolors=n_categories, clip=False)

# x_0,y_0 = np.where(time_lag == 6)
# plt.imshow(time_lag, cmap=custom_cmap, norm=norm)
# plt.scatter(y_0,x_0,marker='.',s=1)


fig = plt.figure(dpi=300)

##### main axis
# left, bottom, width, height = 0, 0.1, 0.96, 0.84  # the position of left lower corner, width and height of the figure
left, bottom, width, height = 0, 0.1, 0.96, 0.80  # the position of left lower corner, width and height of the figure

proj = ccrs.PlateCarree()
ax = fig.add_subplot([left, bottom, width, height], projection=proj)

# set land and ocean features
ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
ax.add_feature(cfeature.OCEAN, zorder=2)
# add amazon basin boundary
add_shp(ax, lw=0.5, ec='k', fc='none', zorder=3)

ax.imshow(time_lag, origin='upper',
          extent=[-79.77497863776252, -44.516103736085974, -20.521922834169924, 8.628408135496414],
          transform=ccrs.PlateCarree(), zorder=2, cmap=custom_cmap, norm=norm, )  # norm=norm,

# 设置显示经纬度范围, 注意指定crs关键字,否则范围不一定完全准确
for tmp_i in np.arange(len(region_list)):
    ax.plot(region_list[tmp_i][1], region_list[tmp_i][0], marker='*', color='k',  #
            alpha=0.9, markersize=12, markeredgecolor='None', zorder=4)  # markeredgecolor='None'

    ax.text(region_list[tmp_i][1] - 0.6, region_list[tmp_i][0] - 2, title_list[tmp_i].split('_')[0],
            color="k", fontsize=10, )  # 半透明黑底白字

# # add the boundary of the example region
# ax.plot(boundary_1_geo[:, 0], boundary_1_geo[:, 1], color='k', zorder=4, linewidth=1.5)  # high asynchrony
# ax.plot(boundary_2_geo[:, 0], boundary_2_geo[:, 1], color='darkred', zorder=4, linewidth=1.5)  # low asynchrony

extents = [-80, -44, -22, 10]
ax.set_extent(extents, crs=proj)

## 设置经纬度major & minor刻度
ax.set_xticks(np.arange(-80, -40, 10), crs=proj)
ax.set_yticks(np.arange(-20, 10 + 10, 10), crs=proj)
## 添加设置网格线
# ax.grid(color=[0.94, 0.94, 0.94], linestyle='--', zorder=1)
## 利用Formatter格式化刻度标签
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

# ax.tick_params(axis='both', labelsize=12, direction='out', colors='k', length=3, width=0.9, which='major',
#                left=False, right=True, top=True, bottom=False,
#                labelleft=False, labelright=True, labeltop=True, labelbottom=False, zorder=10)

ax.tick_params(axis='both', labelsize=12, direction='out', colors='k', length=3, width=0.9, which='major',
               left=True, right=False, top=True, bottom=False,
               labelleft=True, labelright=False, labeltop=True, labelbottom=False, zorder=10)

p = ax.contourf(lon, lat, p_value_map, levels=[0, 0.05, 1],  alpha=0.001, hatches=['...', None],
                colors="none", zorder=4, transform=ccrs.PlateCarree())

# draw a white rectangle around the subplots
left, bottom, width, height = 0.68, 0.145, 0.30, 0.30
ax0 = fig.add_axes([left, bottom, width, height])
plt.Rectangle((left, bottom), width, height, transform=ax0.transAxes, facecolor='white', zorder=3)
ax0.tick_params(axis='both', which='both', colors='white', left=False, right=False, top=False, bottom=False,
                labelleft=False, labelright=False, labeltop=False, labelbottom=False, )

ax0.spines['bottom'].set_color('white')
ax0.spines['top'].set_color('white')
ax0.spines['right'].set_color('white')
ax0.spines['left'].set_color('white')

# plt.axis('off')

# ax0.set_facecolor('white')
# ax0.axis('off')

left, bottom, width, height = 0.70, 0.23, 0.20, 0.20
ax1 = fig.add_axes([left, bottom, width, height])

ax1.set_facecolor('white')

# 使用 sns.histplot 绘制概率直方图
sns.histplot(
    data=time_lag[~np.isnan(time_lag)],
    stat="probability",
    bins=np.arange(-0.5, 7.5, 1),  # 关键：定义 bin 边界，使每个整数为中心
    color='gray',
    alpha=0.6,
    edgecolor='k',
    linewidth=0.25,
    ax=ax1,
    zorder=3
)

# 设置 y 轴范围
ax1.set_ylim(0, 0.30)

# y 轴标签在右侧
ax1.yaxis.set_ticks_position('right')
ax1.yaxis.set_label_position('right')

# 设置 x 轴范围和刻度
ax1.set_xlim(-0.6, 6.6)
ax1.set_xticks(np.arange(0, 7))  # 刻度在整数位置（即每个柱子中心）
ax1.set_xticklabels(np.arange(0, 7), fontsize=10, ha='center')  # 水平居中

# 标签
ax1.set_ylabel('Density', fontsize=10)
ax1.set_xlabel(r'$\triangle$$\mathit{t}$ (month)', fontsize=10, labelpad=0.5)

cbar = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap),
    ax=ax,
    orientation='horizontal',
    fraction=0.04,
    pad=0.03,
    boundaries=bounds,  # 必须指定边界
    ticks=np.arange(0, 7) + 0.5,  # 标签居中
    drawedges=True,  # 显示分隔线（关键！）
    spacing='uniform'  # 确保每段等宽
)

# 设置标签
cbar.set_label(r'$\triangle$$\mathit{t}$ (month)', fontsize=12)
cbar.ax.set_xticklabels(['0', '1', '2', '3', '4', '5', '6'])  # 显示整数标签
cbar.ax.tick_params(length=0)  # 可选：隐藏刻度线，更简洁

# Low time lag for the low asynchrony degree, and high time lag for the high asynchrony degree
ax.text(-78.4, -24, 'Low', clip_on=True, transform=ccrs.PlateCarree())
ax.text(-47.5, -24, 'High', clip_on=True, transform=ccrs.PlateCarree())




# In[] visualize the EVI amplitude map
EVI_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Data\seasonality\BRDF_EVI.tif'  # MAIACI_EVI.tif
_geo_evi, _prj_evi, EVI_amazon = readTif_gdal(EVI_path)
EVI_ampl = np.nanmax(EVI_amazon, axis=2) - np.nanmin(EVI_amazon, axis=2)

EVI_ampl = cv2.resize(EVI_ampl, (raws_y, colums_x))
EVI_ampl[EVI_ampl > 1] = np.nan

EVI_ampl[forest_mask == 0] = np.nan  # mask the non-forest area

# In[] set the colormap for EVI amplitude
cmap = plt.cm.BrBG_r
# norm = mcolors.Normalize(vmin=-1, vmax=500)

light_gray = np.array([0.83, 0.83, 0.83, 1.0])
colors = cmap(np.linspace(0, 1, cmap.N))
# colors[0] = np.array([0.83, 0.83, 0.83, 1.0])  # Set the first color to gray (for the value -2)
custom_cmap = mcolors.ListedColormap(colors)

norm = mcolors.Normalize(vmin=0, vmax=0.2)

# custom_cmap = ListedColormap(custom_cmap(np.linspace(0, 0.1, 256)))
# custom_cmap.set_bad(color=light_gray) # Set the background color to gray
# custom_cmap.set_under(color=light_gray)  # Set the background color to gray

# In[] visualize the EVI amplitude map --- Fig S

fig = plt.figure(dpi=300)

##### main axis
left, bottom, width, height = 0, 0.1, 0.96, 0.80  # the position of left lower corner, width and height of the figure

proj = ccrs.PlateCarree()
ax = fig.add_subplot([left, bottom, width, height], projection=proj)

# set land and ocean features
ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
ax.add_feature(cfeature.OCEAN, zorder=2)
# add amazon basin boundary
add_shp(ax, lw=0.8, ec='k', fc='none', zorder=3)

plt.imshow(EVI_ampl, origin='upper',
           extent=[-79.77497863776252, -44.516103736085974, -20.521922834169924, 8.628408135496414],
           transform=ccrs.PlateCarree(), zorder=2, cmap=custom_cmap, norm=norm)

# 设置显示经纬度范围, 注意指定crs关键字,否则范围不一定完全准确
extents = [-80, -44, -22, 10]
ax.set_extent(extents, crs=proj)

## 设置经纬度major & minor刻度
ax.set_xticks(np.arange(-80, -40, 10), crs=proj)
ax.set_yticks(np.arange(-20, 10 + 10, 10), crs=proj)
## 添加设置网格线
# ax.grid(color=[0.94, 0.94, 0.94], linestyle='--', zorder=1)
## 利用Formatter格式化刻度标签
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.tick_params(axis='both', labelsize=12, direction='out', colors='k', length=3, width=0.9, which='major',
               left=True, right=False, top=True, bottom=False,
               labelleft=True, labelright=False, labeltop=True, labelbottom=False, zorder=10)

# legend_elements = [Patch(facecolor=light_gray, edgecolor='k', label='Non-intact forest')]
# ax.legend(handles=legend_elements, fontsize=11, frameon=False, prop={'size': 12},
#           bbox_to_anchor=(0.48, 0.095))  # loc='lower left',

# draw a white rectangle around the subplots
left, bottom, width, height = 0.680, 0.145, 0.30, 0.30
ax0 = fig.add_axes([left, bottom, width, height])
plt.Rectangle((left, bottom), width, height, transform=ax0.transAxes, facecolor='white', zorder=2)
ax0.tick_params(axis='both', which='both', colors='white', left=False, right=False, top=False, bottom=False,
                labelleft=False, labelright=False, labeltop=False, labelbottom=False, )

ax0.spines['bottom'].set_color('white')
ax0.spines['top'].set_color('white')
ax0.spines['right'].set_color('white')
ax0.spines['left'].set_color('white')

# plt.axis('off')
# ax0.set_facecolor('white')
# ax0.axis('off')

left, bottom, width, height = 0.70, 0.23, 0.20, 0.20
ax1 = fig.add_axes([left, bottom, width, height])

# draw a white rectangle around the subplots
# pos = ax1.get_position()
# rect = plt.Rectangle((pos.x0 - 0.2, pos.y0 - 0.2), 2,2, transform=ax.transAxes, edgecolor='g', facecolor='white', zorder=2)
# ax1.add_patch(rect)

# ax1.figure.patch.set_facecolor('white')

ax1.set_facecolor('white')
# rect = ax1.patch
# rect.set_facecolor('white')
# fig.patch.set_facecolor('gray')
# ax1.hist(delat_r2, bins=50, color='gray', alpha=0.6, edgecolor='k', linewidth=0.25, density=1)
sns.histplot(data=EVI_ampl[~np.isnan(EVI_ampl)], stat="probability",
             bins=50, color='gray', alpha=0.6, edgecolor='k', ax=ax1, zorder=3)  # color='gray'

ax1.set_ylim(0, 0.3)

ax1.yaxis.set_ticks_position('right')
ax1.yaxis.set_label_position('right')

ax1.set_xlim(0, 0.2)
ax1.set_ylabel('Density', fontsize=10)
ax1.set_xlabel('EVI amplitude', fontsize=10, labelpad=0.5)
ax1.xaxis.set_label_coords(0.6, -0.28)

# ax1.vlines(90, 0, 1.0, color='darkred', linewidth=1.0, linestyle='--')
# ax1.text(5, 0.80, '(>90%): {}%'.format(round(percent_ever * 100, 1)), fontsize=10, color='darkred')

# Set the background color behind the ticks and labels
ax1.tick_params(axis='both', which='both', colors='black')
# Set axis labels and title with a white background
# ax1.xaxis.label.set_backgroundcolor('white')
# ax1.yaxis.label.set_backgroundcolor('white')


# legend_elements = [Patch(facecolor=light_gray, edgecolor='k', label='Non-Forest')]
# ax.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=False, prop={'size': 12})

# show the colorbar of the deciduousness amplitude


# show the colorbar of the deciduousness amplitude

# show the colorbar of the deciduousness amplitude
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax, orientation='horizontal',
                    fraction=0.04, pad=0.03)

cbar.ax.set_xlim(0.00, 0.2)

cbar.set_label('EVI amplitude', fontsize=12)

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Figures\EVI_Amplitude_0817.png'
# plt.savefig(save_path, bbox_inches='tight', dpi=300)


# In[] relationship between EVI amplitude and deciduousness amplitude
dec_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Multiyear_mosaic\250m_gapfill\v2_threshold_0.45_overestimate\95qc_DSO_1111\Mean_Filled_95qc\Composite_Data_5km_gf_3y.tif'
_geo, _prj, dec_month = readTif_gdal(dec_path)

dec_month = dec_month.astype(np.float32)
dec_month[dec_month > 1000] = np.nan
dec_month[forest_mask == False] = np.nan

dec_ampl = np.nanmax(dec_month, axis=2) - np.nanmin(dec_month, axis=2)
dec_ampl = dec_ampl / 10.0
# mask of deciduousness amplitude and EVI amplitude
EVI_ampl[EVI_ampl > 0.2] = np.nan  # mask the EVI amplitude larger than 0.2
final_maks = ~np.logical_or(np.isnan(EVI_ampl), np.isnan(dec_ampl))

dec_forest_res = dec_ampl[final_maks]
evi_forest_res = EVI_ampl[final_maks]

# In[] scatter plot the deciduousness amplitude and EVI amplitude -- linear regression
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
plt.hexbin(dec_forest_res, evi_forest_res, gridsize=200, cmap='Greens', mincnt=1, edgecolors='k', linewidths=0.1)
r_2, p_value = stats.pearsonr(dec_forest_res, evi_forest_res)
slope, intercept, r_value, p_value, std_err = stats.linregress(dec_forest_res, evi_forest_res)
x = np.linspace(0, 100, 200)
y = slope * x + intercept
plt.plot(x, y, color='k', linewidth=1)

r2_text = rf'$r = {r_2:.2f}$'
p_text = f'$p < 0.005$'
plt.text(5, 0.18, '$r = %0.2f$' % r_2, fontsize=12, color='k')
plt.text(5, 0.17, '$p$ < 0.005', fontsize=12, color='k')
plt.text(5, 0.16, 'y = %0.3f * x + %0.3f' % (slope, intercept), fontsize=12, color='k')

cbar = plt.colorbar(ax.collections[0], ax=ax, fraction=0.04, pad=0.03)
cbar.set_label('Counts', fontsize=12)

plt.ylabel('EVI amplitude', fontsize=14)
plt.xlabel('Deciduousness amplitude(%)', fontsize=14)
plt.ylim(-0.005, 0.205)
plt.xlim(-1, 101)
plt.tight_layout()

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Figures\EVI_Dec_Amplitude_Relationship_0817.png'
# plt.savefig(save_path, bbox_inches='tight', dpi=300)

# In[] peak EVI timing and deciduousness peak timing visualization
dec_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Multiyear_mosaic\250m_gapfill\v2_threshold_0.45_overestimate\95qc_DSO_1111\Mean_Filled_95qc\Composite_Data_5km_gf_3y.tif'
_geo, _prj, dec_month = readTif_gdal(dec_path)
dec_month = dec_month.astype(np.float32)
dec_month[dec_month > 1000] = np.nan
dec_month[forest_mask == False] = np.nan

rows_y, columns_x = dec_month.shape[:2]  # 正确获取维度

EVI_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Data\seasonality\BRDF_EVI.tif'  # MAIACI_EVI.tif
_geo_evi, _prj_evi, EVI_amazon = readTif_gdal(EVI_path)
EVI_amazon = EVI_amazon.astype(np.float32)

EVI_amazon = cv2.resize(EVI_amazon, (raws_y, colums_x))

dec_mask = np.isnan(dec_month).all(axis=2)
evi_mask = np.isnan(EVI_amazon).all(axis=2)

final_mask = ~np.logical_or(dec_mask, evi_mask)

dec_forest_res = dec_month[final_mask]
evi_forest_res = EVI_amazon[final_mask]
dec_peak_month = np.nanargmax(dec_forest_res, axis=1) + 1
evi_peak_month = np.nanargmax(evi_forest_res, axis=1) + 1

dec_peak_map = np.full((rows_y, columns_x), np.nan)
evi_peak_map = np.full((rows_y, columns_x), np.nan)

# Step 5: 将结果写回空间网格
dec_peak_map[final_mask] = dec_peak_month
evi_peak_map[final_mask] = evi_peak_month

time_lag = evi_peak_map - dec_peak_map

time_lag[time_lag > 6] = 12 - time_lag[time_lag > 6]

plt.imshow(time_lag)



# In[]
r_evi_dec_res = correlation_map.reshape(-1)
r_evi_dec_forest = r_evi_dec_res[~np.isnan(r_evi_dec_res)]
proportion_weak = np.nansum(r_evi_dec_forest > -0.5) / len(r_evi_dec_forest)

# In[] load the numpy data -- for leaf age proposal --JW leaf aga proposal

sos_path = r'D:\OneDrive - The University Of Hong Kong\Proposals\leaf age proposal\Figures\data\ATTO_SOS2020.npy'
eos_path = r'D:\OneDrive - The University Of Hong Kong\Proposals\leaf age proposal\Figures\data\ATTO_EOS2020.npy'

sos = np.load(sos_path)
eos = np.load(eos_path)

# In[] imshow the SOS and EOS map
# import datetime
# datetime.datetime(2020, 1, 1) + datetime.timedelta(150 - 1)
#
# doy_list = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366, 397, 425, 456]
# doy_list = np.array(doy_list)
# xlabel_list = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D', 'J', 'F', 'M', 'A']

# In[]
# cmap = plt.cm.terrain_r
#
# colors = cmap(np.linspace(0, 1, cmap.N))
# custom_cmap = mcolors.ListedColormap(colors)
#
# # norm = mcolors.PowerNorm(gamma=0.8, vmin=150, vmax=250)
# norm = mcolors.Normalize(vmin=140, vmax=250)
#
# fig, ax = plt.subplots(dpi=300, figsize=(7, 6))
#
# plt.imshow(sos, norm=norm, cmap=custom_cmap)
# cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax, orientation='horizontal', fraction=0.04, pad=0.01)
# cbar.ax.tick_params(labelsize=22)
#
# # cbar.set_ticks([152,182,213,244])
# # cbar.set_ticklabels(['Jun','July','Aug.','Sept.'])
#
# cbar.set_label('Day of Year', fontsize=22)
#
# plt.axis('off')
# plt.tight_layout()
#
# save_path = r'D:\OneDrive - The University Of Hong Kong\Proposals\leaf age proposal\Figures\SOS.png'
# plt.savefig(save_path, bbox_inches='tight')

# In[] imshow the EOS map eos[eos < 100] = eos[eos < 100] + 365
cmap = plt.cm.terrain_r

colors = cmap(np.linspace(0, 1, cmap.N))
custom_cmap = mcolors.ListedColormap(colors)

# norm = mcolors.PowerNorm(gamma=0.8, vmin=150, vmax=250)
norm = mcolors.Normalize(vmin=0, vmax=365)

np.nanmin(eos), np.nanmax(eos)

fig, ax = plt.subplots(dpi=300)

plt.imshow(eos, norm=norm, cmap=custom_cmap)
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax, orientation='horizontal', fraction=0.04,
                    pad=0.01)
cbar.ax.tick_params(labelsize=22)
cbar.set_ticks([0, 100, 200, 300])
cbar.set_label('Day of Year', fontsize=22)

plt.axis('off')
plt.tight_layout()

save_path = r'D:\OneDrive - The University Of Hong Kong\Proposals\leaf age proposal\Figures\EOS.png'

# plt.savefig(save_path,bbox_inches='tight')


# In[]
import pandas as pd

pd_file_path = r'D:\OneDrive - The University Of Hong Kong\Proposals\leaf age proposal\Figures\data\Leaf_age_evaluation.csv'

pd_data = pd.read_csv(pd_file_path)

# In[] Plot
fig, ax = plt.subplots(figsize=(4, 5), dpi=300)

# Set bar height
bar_height = 0.2
sites = pd_data['Site'].tolist()
# Set positions for each bar group
r1 = np.arange(len(sites))
r2 = [x + bar_height for x in r1]
r3 = [x + bar_height for x in r2]

# Create horizontal bars
ax.barh(r1, pd_data['Young'].tolist(), color=(144 / 255, 238 / 255, 144 / 255), height=bar_height, edgecolor='grey',
        label='Young')
ax.barh(r2, pd_data['Mature'].tolist(), color=(1 / 255, 128 / 255, 0), height=bar_height, edgecolor='grey',
        label='Mature')
ax.barh(r3, pd_data['Old'].tolist(), color=(152 / 255, 51 / 255, 0), height=bar_height, edgecolor='grey', label='Old')

ax.set_xlim(0, 1)

ax.set_xticks([0, 0.2, 0.40, 0.6, 0.8, 1])
ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'], fontsize=14)

# Add labels
# ax.set_ylabel('Site')
ax.set_xlabel('Correlation Coefficient', fontsize=16)
ax.set_yticks([r + bar_height for r in range(len(sites))])
ax.set_yticklabels(sites, fontsize=14)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# ax.xaxis.tick_top()
# ax.xaxis.set_label_position('top')

# Add legend
# ax.legend(bbox_to_anchor=(1.1, 0.2))

plt.tight_layout()

# Show plot
