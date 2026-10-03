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
from matplotlib.ticker import StrMethodFormatter

plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams["axes.unicode_minus"] = True  # 显示负号
matplotlib.use('Qt5Agg')
matplotlib.rcParams['figure.dpi'] = 300

from numpy.lib.stride_tricks import sliding_window_view


# In[]

def readTif_gdal(fileName, nbands=36):
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    dataset = gdal.Open(fileName)
    if dataset == None:
        print("cannot open file:" + fileName)
        return
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    # transpose
    if im_data.shape[0] <= nbands:
        im_data = np.transpose(im_data, [1, 2, 0])
    return dataset.GetGeoTransform(), dataset.GetProjection(), im_data


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


# using the max r to across the certain window size to avoid the potential noise in the fine resolution
def find_max_r_timeseries_custom(image_series, target_series, center_x, center_y,
                                 search_radius=1, window_size=2):
    """
    在自定义区域内滑动窗口，计算窗口时序与目标时序的相关系数，返回最大r对应的时序
    参数:
        image_series: 图像时序数据 (W, H, T) 格式的NumPy数组
        target_series: 目标时序数据 (T,) 一维数组
        center_x: 中心像素的x坐标 (宽度方向)
        center_y: 中心像素的y坐标 (高度方向)
        search_radius: 搜索区域半径 (默认1，即3x3区域)
        window_size: 滑动窗口大小 (默认2，即2x2窗口)
    返回:
        max_r: 最大相关系数
        best_window_series: 对应窗口的时序数据 (window_size, window_size, T)
        best_window_pos: 窗口左上角坐标 (x, y)
    """
    # 检查输入维度
    if image_series.ndim != 3:
        raise ValueError("image_series应为(W,H,T)格式")

    # 获取图像尺寸和时间步
    w, h, t = image_series.shape
    if len(target_series) != t:
        raise ValueError("目标时序长度与图像时序长度不匹配")

    # 计算搜索区域边界
    x_start = max(0, center_x - search_radius)
    x_end = min(w, center_x + search_radius + 1)  # +1因为Python切片是左闭右开
    y_start = max(0, center_y - search_radius)
    y_end = min(h, center_y + search_radius + 1)

    # 提取搜索区域的时序数据
    region_series = image_series[x_start:x_end, y_start:y_end, :]

    if np.nanmax(region_series) > 100:
        region_series = region_series / 1000.0

    # 检查区域是否足够形成指定窗口
    if region_series.shape[0] < window_size or region_series.shape[1] < window_size:
        # 返回中心像素
        center_in_region_x = min(search_radius, center_x - x_start)
        center_in_region_y = min(search_radius, center_y - y_start)
        window_series = region_series[
                        center_in_region_x:center_in_region_x + 1,
                        center_in_region_y:center_in_region_y + 1, :]
        window_series = np.expand_dims(window_series, axis=(0, 1))  # 保持(1,1,T)形状

        # 计算相关系数
        r = np.corrcoef(window_series.reshape(t, -1).T,
                        target_series.reshape(1, -1))[0, 1]

        # 计算rmse
        rmse = np.sqrt(np.mean((window_series.reshape(t, -1) - target_series.reshape(1, -1)) ** 2))

        return r, rmse, window_series, (x_start, y_start)

    # 创建所有可能的滑动窗口视图
    windows = sliding_window_view(region_series, (window_size, window_size, t), axis=(0, 1, 2))
    windows = windows[:, :, 0]  # 因为时间维度已经固定

    max_r = -1
    best_window_series = None
    best_window_pos = (0, 0)

    # 遍历所有窗口
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            window_series = windows[i, j]  # 形状为(window_size, window_size, T)

            # 展平窗口的空间维度，保留时间维度
            window_flat = window_series.reshape(-1, t)  # (window_size*window_size, T)
            #
            # window_flat = np.nanmean(window_flat, axis=0)  # 平均化窗口内的时序

            # 计算与目标时序的相关系数矩阵
            corr_matrix = np.corrcoef(window_flat, target_series.reshape(1, -1))
            errors = window_flat - target_series.reshape(1, -1)

            rmse_values = np.sqrt(np.mean(errors ** 2, axis=1))  # 各像素的RMSE
            rmse_values = np.nanmean(rmse_values)  # 平均化窗口内的RMSE

            # print(rmse_values.shape)

            # 取窗口内所有像素时序与目标时序的平均相关系数
            r = np.mean(corr_matrix[:-1, -1]) - 1.5 * rmse_values  # [0]  # 最后一行/列是目标时序

            if r > max_r:
                max_r = r
                best_window_series = window_series
                best_window_pos = (x_start + i, y_start + j)

    return max_r, best_window_series, best_window_pos


# In[] MODIS forest mask
# raws_y, columns_x = 786, 650
raws_y, columns_x = 15724, 13004
cls_modis_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig1\Data\Forest_Mask\MCD12Q1_Amazon_500m.tif'
# cls_modis_path = r'X:\Song_Amazon_Mapping\classification_map.tif'
_, _, cls_md = readTif_gdal(cls_modis_path)

cls_md_forest = copy.deepcopy(cls_md)
cls_md_forest = cls_md_forest.astype(np.float32)

cls_md_forest[cls_md_forest != 2] = -1
cls_md_forest[cls_md == 0] = np.nan

cls_modis_coarse = cv2.resize(cls_md_forest, (raws_y, columns_x), interpolation=cv2.INTER_NEAREST)
forest_mask = copy.deepcopy(cls_modis_coarse == 2)

# remove the small noise and open the small holes
kernel = morphology.disk(1)
mis_pixmph = morphology.opening(forest_mask, kernel)
forest_mask = morphology.remove_small_objects(mis_pixmph, 256, connectivity=1)
kernel = morphology.disk(1)
forest_mask = morphology.opening(forest_mask, kernel)

# In[] FigS4 temporal level evaluation with phenocam evaluation

phencoam_site = ['BCI', 'K34', 'K67', 'ATTO', 'RJA']
phenocam_dir = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Phenocam_GCC\Deciduousness'

# extract phenocam seasonality
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

# In[]
# extract the dec_month value at the phenocam site based on the lat and lon
# dec_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Multiyear_mosaic\250m_gapfill\v2_threshold_0.45_overestimate\95qc_DSO_1111\Mean_Filled_95qc\Composite_Data_5km_gf_3y.tif'
# _geo, _prj, dec_month = readTif_gdal(dec_path)
# dec_month = dec_month.astype(np.float32)
# dec_month[dec_month > 1000] = np.nan
# dec_month[~forest_mask] = np.nan
# dec_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Multiyear_mosaic\250m_gapfill\v2_threshold_0.45_overestimate\95qc_DSO_1111\Mean_Filled_95qc\Composite_Data_5km_gf_3y.tif'
dec_path = r'D:\OneDrive - The University Of Hong Kong\PhD_Projects\Project4_Mapping_Amazon_Basin_Using_GEE\Composite_Data_250m_gf_3y.tif'

# dec_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Supplementary_Figures\Composite_Data_5km_gf_3y_FFT_one_components.tif'
_geo, _prj, dec_month = readTif_gdal(dec_path)

dec_month = dec_month.astype(np.float32)
dec_month[dec_month > 1000] = np.nan
dec_month[~forest_mask] = np.nan

location_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Phenocam_GCC\ATTO_RJA_Location.csv'
loc_df = pd.read_csv(location_path)

# In[] extract the dec_seasonality

target_list = phenocam_value_list[1:]
target_site_list = phencoam_site[1:]
# extract the dec_month value at the phenocam site based on the lat and lon
dec_value_list = []

# fig, axes = plt.subplots(2, 2, figsize=(6, 5))

# ax_flatten = axes.flatten()

for i, site_name in enumerate(target_site_list):
    # select the location based on the site_name
    site_df = loc_df[loc_df['Site'] == site_name]
    site_lat = site_df['Lat'].to_numpy()[0]
    site_lon = site_df['Lon'].to_numpy()[0]
    # extract the dec_month value at the phenocam site based on the lat and lon
    dec_y, dec_x = int((site_lon - _geo[0]) / _geo[1]), int((site_lat - _geo[3]) / _geo[5])

    phenocam_seasonality = target_list[i]
    search_radius = 10  # (radius=2 → 2 * radius + 1=5)
    window_size = 2  # 使用3x3窗口
    # 计算并获取结果
    max_r, best_series, best_pos = find_max_r_timeseries_custom(
        dec_month, phenocam_seasonality, dec_x, dec_y,
        search_radius=search_radius, window_size=window_size
    )
    dec_site_list = np.nanmean(best_series, axis=(0, 1))  # 平均化窗口内的时序
    dec_value_list.append(dec_site_list)
    #
    # ax_flatten[i].plot(np.arange(1, 13, 1), dec_site_list , marker='o', label='Sentinel-2', markersize=2,
    #                    linewidth=1)
    # ax_flatten[i].plot(np.arange(1, 13, 1), phenocam_seasonality, marker='*', color='k', label='Phenocam', markersize=2,
    #                    linewidth=1)
    # ax_flatten[i].set_xticks(np.arange(1, 13, 1))
    # ax_flatten[i].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    # ax_flatten[i].set_xlabel('Month', fontsize=6)
    # ax_flatten[i].set_ylabel('Deciduousness(%)', fontsize=6)
    # ax_flatten[i].tick_params(axis='both', which='major', labelsize=6, length=2)
    # ax_flatten[i].set_ylim(0, 0.5)
    # ax_flatten[i].set_title(site_name, fontsize=6)

    # print(site_name, 'Max r:', max_r)

# In[]
BCI_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Phenocam_GCC\Deciduousness\BCI_Sentinel-2_Dec.csv'
BCI_df = pd.read_csv(BCI_path)
BCI_dec_monthly = BCI_df['Deciduous'].values

satellite_dec_list = [BCI_dec_monthly]
satellite_dec_list.extend(dec_value_list)


# In[] seasonality visulization

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
    r2 = np.corrcoef(phenocam_value_list[i], satellite_dec_list[i])[0, 1] #** 2
    rmse = np.sqrt(np.mean((phenocam_value_list[i] - satellite_dec_list[i]) ** 2))

    r_list.append(r2)
    rmse_list.append(rmse)

    dec_values_list.extend(satellite_dec_list[i])
    phenocam_all_list.extend(phenocam_value_list[i])

    ax = axes.flatten()[i]
    lns1 = ax.plot(month_x, phenocam_value_list[i] * 100, marker='o', color='k', label='Phenocam', markersize=2,
                   linewidth=1)
    # ax1 = ax.twinx()
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
    # ax1.tick_params(axis='both', which='major', labelsize=6, length=2)
    ax.set_xlabel('Month', fontsize=7)
    ax.set_ylabel('Deciduousness(%)', fontsize=7)
    ax.set_ylim(-2, 50)

    # ax.set_ylabel('Phenocam Deciduousness (%)', fontsize=7)
    # ax1.set_ylabel('Sentinel-2 Deciduousness (%)', fontsize=7)

    if site_name == 'ATTO':
        ax.legend(loc='upper right', fontsize=6, frameon=False)

# Show the scatter plot of the phenocam and satellite data

ax = axes.flatten()[-1]
# plt.scatter(dec_values_list2, litterfall_values_list, s=30, alpha=0.6)

for i in range(len(site_name_list)):
    ax.scatter(np.array(satellite_dec_list[i]) * 100, phenocam_value_list[i] * 100, s=7, alpha=0.9,
               label=site_name_list[i].split('-')[0])

r_dec_phenocam = np.corrcoef(phenocam_all_list, dec_values_list)[0, 1]
rmse_phenocam = np.sqrt(np.mean((np.array(phenocam_all_list) - np.array(dec_values_list)) ** 2))


# linear fit
def linear_func(x, a, b):
    return a * x + b


y = np.array(phenocam_all_list)
x = np.array(dec_values_list)
popt, pcov = curve_fit(linear_func, x, y)
x_fit = np.linspace(min(x) * 100, max(x) * 100, 100)
y_fit = linear_func(x_fit, *popt)
ax.plot(x_fit, y_fit, color='k', linewidth=0.8)
# equation
a, b = popt
# p value
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ax.text(0.05, 0.90, 'y = %.2f*x + %.2f' % (slope, intercept), fontsize=7, transform=plt.gca().transAxes)
ax.text(0.05, 0.80, '$r$ = %.2f' % r_value, fontsize=7, transform=plt.gca().transAxes)
ax.text(0.05, 0.70, 'RMSE = %.2f' % rmse_phenocam, fontsize=7, transform=plt.gca().transAxes)
p_exp = int(np.floor(np.log10(p_value)))
p_coeff = p_value / (10 ** p_exp)

ax.text(0.05, 0.60,rf'$p$ = {p_coeff:.2f} $\times$ 10$^{{{p_exp}}}$',
    fontsize=7,transform=plt.gca().transAxes
)
# ax.text(0.05, 0.60, 'p < 0.001', fontsize=7, transform=plt.gca().transAxes)

ax.set_xlabel('Sentinel-2 derived deciduousness(%)', fontsize=7)
ax.set_ylabel('Phenocam derived deciduousness(%)', fontsize=7)

ax.set_xlim(-5, 45)
ax.set_ylim(-5, 45)

ax.tick_params(axis='both', which='major', labelsize=6, length=2)
ax.legend(loc='lower right', fontsize=6, frameon=True, ncol=1)

plt.show()

# # 产生figure legend
# lns = lns1 + lns2
# labs = [l.get_label() for l in lns]
# fig.legend(lns, labs, fontsize=6, frameon=False, ncol=2, bbox_to_anchor=(0.6, 0.05))  # , )

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Nature_Revision_Round1\FigS_phenocam_validataion.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300)

print('mean r: %.2f' % np.mean(r_list))
print('rmse: %.2f' % np.mean(rmse_list))

# In[] Phenocam seasonality evaluation with the deciduousness fraction only for two-sites

phencoam_site = ['ATTO', 'RJA']
phenocam_dir = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Phenocam_GCC\Deciduousness'

# extract phenocam seasonality
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
    phenocam_value_list.append(phenocam_seasonality)

# extract the dec_month value at the phenocam site based on the lat and lon
dec_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Multiyear_mosaic\250m_gapfill\v2_threshold_0.45_overestimate\95qc_DSO_1111\Mean_Filled_95qc\Composite_Data_5km_gf_3y.tif'
_geo, _prj, dec_month = readTif_gdal(dec_path)

dec_month = dec_month.astype(np.float32)
dec_month[dec_month > 1000] = np.nan

dec_month[~forest_mask] = np.nan

location_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Phenocam_GCC\ATTO_RJA_Location.csv'
loc_df = pd.read_csv(location_path)

ATTO_lat = loc_df['Lat'].to_numpy()[0]
ATTO_lon = loc_df['Lon'].to_numpy()[0]

RJA_lat = loc_df['Lat'].to_numpy()[1]
RJA_lon = loc_df['Lon'].to_numpy()[1]

# extract the EVI value at the phenocam site based on the lat and lon -- ATTO, RJA site
ATTO_y, ATTO_x = int((ATTO_lon - _geo[0]) / _geo[1]), int((ATTO_lat - _geo[3]) / _geo[5])
RJA_y, RJA_x = int((RJA_lon - _geo[0]) / _geo[1]), int((RJA_lat - _geo[3]) / _geo[5])

patch_size = 1
ATTO_EVI = np.nanmean(
    dec_month[ATTO_x - patch_size:ATTO_x + patch_size + 1, ATTO_y - patch_size:ATTO_y + patch_size + 1],
    axis=(0, 1)) / 10
RJA_EVI = np.nanmean(dec_month[RJA_x - patch_size:RJA_x + patch_size + 1, RJA_y - patch_size:RJA_y + patch_size + 1],
                     axis=(0, 1)) / 10

satellite_dec_list = [ATTO_EVI, RJA_EVI]

# In[]
site_name_list = ['ATTO', 'RJA']
month_label = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.8))
plt.subplots_adjust(wspace=0.6, hspace=0.3, left=0.1, right=0.9, top=0.95, bottom=0.2)

for i in np.arange(len(axes.flatten())):
    site_name = site_name_list[i]

    month_x = np.arange(1, 13, 1)
    r2 = np.corrcoef(phenocam_value_list[i], satellite_dec_list[i])[0, 1] ** 2
    ax = axes.flatten()[i]
    lns1 = ax.plot(month_x, phenocam_value_list[i] * 100, marker='o', color='k', label='Phenocam', markersize=4,
                   linewidth=1)
    ax1 = ax.twinx()
    lns2 = ax1.plot(np.arange(1, 13, 1), satellite_dec_list[i], marker='o', label='Sentinel-2', markersize=4,
                    linewidth=1)

    ax.text(0.05, 0.85, site_name, fontsize=10, transform=ax.transAxes)
    ax.text(0.05, 0.75, 'r$^2$ = %.2f' % r2, fontsize=10, transform=ax.transAxes)

    ax.set_xticks(np.arange(1, 13, 1))
    ax.set_xticklabels(month_label)
    ax.tick_params(axis='both', which='major', labelsize=10, length=2)
    ax1.tick_params(axis='both', which='major', labelsize=10, length=2)

    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('Phenocam deciduousness(%)', fontsize=10)
    ax1.set_ylabel('Sentinel-2 deciduousness(%)', fontsize=10)

# 产生figure legend
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
fig.legend(lns, labs, fontsize=10, frameon=False, ncol=1, bbox_to_anchor=(0.65, 0.10))  # , )

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Supplementary_Figures\FigS4_Phenocam_Seasonality_Evaluation_1x2.png'
# plt.savefig(save_path, bbox_inches='tight', dpi=300)
