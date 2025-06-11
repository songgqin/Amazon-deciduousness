import os
import copy
import cv2
import numpy as np
from osgeo import gdal
from osgeo import ogr
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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import rasterio
import rasterio.mask
import scipy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# 导入底图包
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.wkt import loads as load_wkt

matplotlib.use('Qt5Agg')
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['pdf.fonttype'] = 42

plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams["axes.unicode_minus"] = True  # 显示负号


# In[]
def add_shp(ax, **kwargs):
    '''
    在地图上画出Amazon边界的shapefile.
    Parameters
    ----------
    ax : GeoAxes
        目标地图.
    **kwargs
        绘制shape时用到的参数.例如linewidth,edgecolor和facecolor等.
    '''
    proj = ccrs.PlateCarree()
    reader = shpreader.Reader(r'..\Data\Amazon_ThreeRegions_Clip.shp')
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


# In[] forest mask of Amazon
raws_y, columns_x = 786, 650

cls_modis_path = r'.\Data\Forest_Mask\MCD12Q1_Amazon.tif'
_, _, cls_md = readTif_gdal(cls_modis_path)

cls_md_forest = copy.deepcopy(cls_md)
cls_md_forest = cls_md_forest.astype(np.float32)

# cls_md_forest[cls_md_forest != 2] = -1
# cls_md_forest[cls_md == 0] = np.nan
cls_md_forest[cls_md_forest != 2] = np.nan
cls_modis_coarse = cv2.resize(cls_md_forest, (raws_y, columns_x), interpolation=cv2.INTER_NEAREST)
forest_mask = copy.deepcopy(cls_modis_coarse)

# valid evergreen forest pixels
ind_where = np.where(forest_mask == 2)
print('evergreen forest pixels:', len(ind_where[0]))

# In[]

raws_y, columns_x = 786, 650
per_cls_path = r'..\Data\Cls_Percent_Amazon_Forest.tif'

_geo, _prj, percent_cls = readTif_gdal(per_cls_path)
percent_cls_amazon = percent_cls[:, :, 2] # the evergreen forest percentage in Amazon basin
percent_cls_amazon[percent_cls_amazon == 9999.0] = np.nan
percent_cls_amazon = cv2.resize(percent_cls_amazon, (raws_y, columns_x))

# In[]

dec_path = r'.\Data\Deciduousness_Seasonality_Amazon.tif'
_geo, _prj, dec_month = readTif_gdal(dec_path)

dec_month = dec_month.astype(np.float32)
dec_month[dec_month > 1000] = np.nan

ampl_dec = np.nanmax(dec_month, axis=2) - np.nanmin(dec_month, axis=2)
ampl_dec = ampl_dec / 10 # convert to percentage

ampl_dec[np.isnan(forest_mask)] = np.nan  # the data outside the Amazon basin is set to nan
#
# print(len(ampl_dec[~np.isnan(ampl_dec)]))
#
# # ampl_dec =ampl_dec *100

# In[] imshow the deciduousness amplitude and evergreen forest percentage

cmap = plt.cm.BrBG_r
# norm = mcolors.Normalize(vmin=-1, vmax=50)
light_gray = np.array([0.83, 0.83, 0.83, 1.0])
colors = cmap(np.linspace(0, 1, cmap.N))
colors[0] = np.array([0.83, 0.83, 0.83, 1.0])  # Set the first color to gray (for the value -2)
custom_cmap = mcolors.ListedColormap(colors)
norm = mcolors.PowerNorm(gamma=1.0, vmin=0, vmax=50)

# In[]

# Create a base colormap from BrBG_r
base_cmap = plt.cm.BrBG_r

# Define the custom colormap with the green gradient from 0 to 10 and brown from 10 to 50
colors = [
    (0 / 50, base_cmap(0.15)),  # Start with green at 0
    (15 / 50, base_cmap(0.50)),  # Transition to brown at 10
    (50 / 50, base_cmap(0.9))  # Full brown by 50
]

# Create the custom colormap
custom_cmap = LinearSegmentedColormap.from_list('custom_brbg', colors, N=256)
# Normalize the data to the range [0, 50]
norm = mcolors.Normalize(vmin=0, vmax=50)
custom_cmap = ListedColormap(custom_cmap(np.linspace(0, 1, 256)))
# custom_cmap.set_bad(color=light_gray) # Set the background color to gray
custom_cmap.set_under(color=light_gray)  # Set the background color to gray

ampl_dec_res_forest = copy.deepcopy(ampl_dec[forest_mask == 2])  # forest mask
ampl_dec_res_forest = ampl_dec_res_forest[~np.isnan(ampl_dec_res_forest)]
percent_dec = np.sum(ampl_dec_res_forest > 10) / len(ampl_dec_res_forest)  # the percentage of deciduousness amplitude > 10%

# In[] Bar plot of percent_dec across different thresholds from 5 to 50 with 5% interval --FigS5
percent_dec_list = []
threshold_list = np.arange(5, 55, 5)
for i in range(5, 55, 5):
    tmp_percent_dec = np.sum(ampl_dec_res_forest > i) / len(ampl_dec_res_forest)
    percent_dec_list.append(tmp_percent_dec)

percent_dec_list = np.array(percent_dec_list)*100
plt.figure(dpi=300)
plt.bar(threshold_list, percent_dec_list, width=4.8, color='grey', alpha=0.6, edgecolor='k', linewidth=0.25)

for i, v in enumerate(percent_dec_list):
    plt.text(threshold_list[i], v + 1, '{:.1f}%'.format(v ), ha='center', fontsize=10)

plt.xlabel('Deciduousness amplitude threshold(%)', fontsize=14)
plt.ylabel('Percentage of deciduousness amplitude \n > threshold(%)', fontsize=14)
plt.ylim(0, 100)

save_path = r'.\Supplementary_Figures\FigSX_Deciduousness_Amplitude_Threshold_Bar.png'
# plt.savefig(save_path, bbox_inches='tight', dpi=300)



# In[] imshow the deciduousness amplitude map
fig = plt.figure(dpi=300)
##### main axis
left, bottom, width, height = 0, 0.1, 0.96, 0.80  # the position of left lower corner, width and height of the figure

proj = ccrs.PlateCarree()

ax = fig.add_subplot([left, bottom, width, height], projection=proj)  # [left, bottom, width, height],

# set land and ocean features
ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
ax.add_feature(cfeature.OCEAN, zorder=2)
# add amazon basin boundary
# add_shp(ax, lw=0.5, ec='k', fc='none', zorder=3)

# add amazon basin boundary
add_shp(ax, lw=0.8, ec='k', fc='none', zorder=5)


plt.imshow(ampl_dec, origin='upper',
           extent=[-79.77497863776252, -44.516103736085974, -20.521922834169924, 8.628408135496414],
           transform=ccrs.PlateCarree(), zorder=4, cmap=custom_cmap, norm=norm)  # ampl_dec_masked

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
sns.histplot(data=ampl_dec_res_forest, stat="probability", bins=50, color='gray', alpha=0.6, edgecolor='k',
             linewidth=0.25, ax=ax1, zorder=3)  # color='gray'

ax1.set_ylim(0, 0.15)

ax1.yaxis.set_ticks_position('right')
ax1.yaxis.set_label_position('right')

ax1.set_xlim(0, 50)
ax1.set_ylabel('Density', fontsize=10)
ax1.set_xlabel('Deciduousness amplitude(%)', fontsize=10, labelpad=0.5)
ax1.xaxis.set_label_coords(0.6, -0.28)

ax1.vlines(10, 0, 0.2, color='darkred', linewidth=1.0, linestyle='--')
ax1.text(12, 0.12, '(>10%): {}%'.format(round(percent_dec * 100, 1)), fontsize=10, color='darkred')

# Set the background color behind the ticks and labels
ax1.tick_params(axis='both', which='both', colors='black')

# Set axis labels and title with a white background
# ax1.xaxis.label.set_backgroundcolor('white')
# ax1.yaxis.label.set_backgroundcolor('white')


# legend_elements = [Patch(facecolor=light_gray, edgecolor='k', label='Non-Forest')]
# ax.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=False, prop={'size': 12})

# show the colorbar of the deciduousness amplitude


# show the colorbar of the deciduousness amplitude
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax, orientation='horizontal',
                    fraction=0.04, pad=0.03)

cbar.ax.set_xlim(0, 50)

cbar.set_label('Deciduousness amplitude(%)', fontsize=12)

save_path = r'..\Main_Figures\Fig1\Deciduousness_Amplitude_Map_0521ng'


# In[] imshow the evergreen precentage map
# percent_cls_amazon[forest_mask == -1] = -1
percent_cls_amazon[np.isnan(forest_mask)] = np.nan
cmap = plt.cm.BrBG
# norm = mcolors.Normalize(vmin=-1, vmax=500)

light_gray = np.array([0.83, 0.83, 0.83, 1.0])
colors = cmap(np.linspace(0, 1, cmap.N))
colors[0] = np.array([0.83, 0.83, 0.83, 1.0])  # Set the first color to gray (for the value -2)
custom_cmap = mcolors.ListedColormap(colors)
norm = mcolors.PowerNorm(gamma=1, vmin=0, vmax=100)

# In[]

# percent_cls_amazon[forest_mask == -1] = -1

# Create a base colormap from BrBG_r
base_cmap = plt.cm.BrBG

# Define the custom colormap with the green gradient from 0 to 10 and brown from 10 to 50
colors = [
    (0 / 100, base_cmap(0.10)),  # Start with green at 0
    (85 / 100, base_cmap(0.50)),  # Transition to brown at 10
    (100 / 100, base_cmap(0.9))  # Full brown by 50
]

# Create the custom colormap
custom_cmap = LinearSegmentedColormap.from_list('custom_brbg', colors, N=256)

# Normalize the data to the range [0, 50]
norm = mcolors.Normalize(vmin=0, vmax=100)

custom_cmap = ListedColormap(custom_cmap(np.linspace(0, 1, 256)))
# custom_cmap.set_bad(color=light_gray) # Set the background color to gray
custom_cmap.set_under(color=light_gray)  # Set the background color to gray

# percent_evergreen = copy.deepcopy(percent_cls_amazon[forest_mask == 2])  # forest mask
percent_evergreen = copy.deepcopy(percent_cls_amazon)  # forest mask
percent_evergreen = percent_evergreen[~np.isnan(ampl_dec)] # ensure the same mask as the deciduousness amplitude
percent_ever = np.nansum(percent_evergreen > 90) / len(percent_evergreen)

# In[] bar plot of percent evergreen across different thresholds from 50 to 100 with 5% interval
percent_ever_list = []
threshold_list = np.arange(50, 100, 5)

for i in range(50, 100, 5):
    tmp_percent_ever = np.nansum(percent_evergreen > i) / len(percent_evergreen)
    percent_ever_list.append(tmp_percent_ever)
percent_ever_list = np.array(percent_ever_list) * 100


plt.figure(dpi=300)
plt.bar(threshold_list, percent_ever_list, width=4.8, color='grey', alpha=0.6, edgecolor='k', linewidth=0.25)
for i, v in enumerate(percent_ever_list):
    plt.text(threshold_list[i], v + 1, '{:.1f}%'.format(v), ha='center', fontsize=10)
plt.xlabel('Evergreen forest cover threshold(%)', fontsize=14)
plt.ylabel('Percentage of evergreen forest cover \n > threshold(%)', fontsize=14)

save_path = r'..\Main_Figures\Supplementary_Figures\FigSX_Evergreen_Cover_Threshold_Bar.png'
# plt.savefig(save_path, bbox_inches='tight', dpi=300)

# In[]

fig = plt.figure(dpi=300)
##### main axis
left, bottom, width, height = 0, 0.1, 0.96, 0.80  # the position of left lower corner, width and height of the figure

proj = ccrs.PlateCarree()
ax = fig.add_subplot([left, bottom, width, height], projection=proj)

# set land and ocean features
ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
ax.add_feature(cfeature.OCEAN, zorder=2)
# add amazon basin boundary
add_shp(ax, lw=0.8, ec='#FFC20A', fc='none', zorder=3)

plt.imshow(percent_cls_amazon, origin='upper',
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
#
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

ax1.set_facecolor('white')

sns.histplot(data=percent_evergreen, stat="probability",
             bins=50, color='gray', alpha=0.6, edgecolor='k', linewidth=0.25, ax=ax1, zorder=3)  # color='gray'

ax1.set_ylim(0, 1.0)

ax1.yaxis.set_ticks_position('right')
ax1.yaxis.set_label_position('right')

ax1.set_xlim(0, 105)
ax1.set_ylabel('Density', fontsize=10)
ax1.set_xlabel('Evergreen forest cover (%)', fontsize=10, labelpad=0.5)
ax1.xaxis.set_label_coords(0.6, -0.28)

ax1.vlines(90, 0, 1.0, color='darkred', linewidth=1.0, linestyle='--')
ax1.text(5, 0.80, '(>90%): {}%'.format(round(percent_ever * 100, 1)), fontsize=10, color='darkred')

# Set the background color behind the ticks and labels
ax1.tick_params(axis='both', which='both', colors='black')
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax, orientation='horizontal',
                    fraction=0.04, pad=0.03)
cbar.ax.set_xlim(0, 100)

cbar.set_label('Evergreen forests cover(%)', fontsize=12)

save_path = r'..\Main_Figures\Fig1\Evergreen_Map_0521.png'
# plt.savefig(save_path, bbox_inches='tight', dpi=300)
