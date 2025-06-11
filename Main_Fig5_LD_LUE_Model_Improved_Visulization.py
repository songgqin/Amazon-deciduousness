import os
import copy
import cv2
import numpy as np
from osgeo import gdal, ogr
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

plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams["axes.unicode_minus"] = True  # 显示负号
matplotlib.use('Qt5Agg')
matplotlib.rcParams['figure.dpi'] = 300


# In[]
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


# In[] Forest mask
raws_y, columns_x = 786, 650

cls_modis_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig1\Data\Forest_Mask\MCD12Q1_Amazon.tif'
# cls_modis_path = r'X:\Song_Amazon_Mapping\classification_map.tif'
_, _, cls_md = readTif_gdal(cls_modis_path)

cls_md_forest = copy.deepcopy(cls_md)
cls_md_forest = cls_md_forest.astype(np.float32)

cls_md_forest[cls_md_forest != 2] = -1
cls_md_forest[cls_md == 0] = np.nan

cls_modis_coarse = cv2.resize(cls_md_forest, (raws_y, columns_x), interpolation=cv2.INTER_NEAREST)

forest_mask = copy.deepcopy(cls_modis_coarse==2)

# print(len(forest_mask[~np.isnan(forest_mask)]))
print(len(np.where(forest_mask == 1)[0]))

forest_mask_cls = copy.deepcopy(forest_mask)

# In[] Improved GPP file

improved_gpp_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Delta_r2_LA_Ensemble_Two_SIF_basin.tif'
_geo, _prj, improved_gpp = readTif_gdal(improved_gpp_path)

delat_r2 = improved_gpp[forest_mask_cls]

time_lag_path =  r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig2\Python-Based Figures\Tif_Data\time_lag_map_0521.tif'
_geo, _prj, time_lag = readTif_gdal(time_lag_path)

asynchrony = time_lag[forest_mask_cls]

joint_mask = np.logical_and(~np.isnan(delat_r2), ~np.isnan(asynchrony))

rmse_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Delta_rmse_LA_Ensemble_Two_SIF_basin.tif'
_geo, _prj, rmse = readTif_gdal(rmse_path)
rmse_basin = rmse[forest_mask_cls]

delat_r2 = delat_r2[joint_mask]
asynchrony = asynchrony[joint_mask]

rmse_basin = rmse_basin[joint_mask]

# rmse_basin[rmse_basin > 1] = 1
rmse_basin = rmse_basin * 100
rmse = rmse * 100

percent_improved_gpp = np.nansum(delat_r2 > 0) / len(delat_r2)

mean_r2 = np.nanmean(delat_r2)
median_r2 = np.nanmedian(delat_r2)

percent_rmse = np.nansum(rmse_basin < 0) / len(rmse_basin)

ampl_dec_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Multiyear_mosaic\250m_gapfill\v2_threshold_0.45_overestimate\95qc_DSO_1111\Mean_Filled_95qc\Composite_Data_5km_gf_3y_amplitude.tif'

_geo, _prj, ampl_dec = readTif_gdal(ampl_dec_path)

ampl_dec = ampl_dec.astype(np.float32)
ampl_dec[ampl_dec > 1000] = np.nan

# In[] obtain the eddy flux location
eddy_flux_name_list = ['K34_CfluxBF_DOY.csv', 'K67_CfluxBF_DOY.csv', 'CAX_CfluxBF_DOY.csv', 'RJA_CfluxBF_DOY.csv']
site_locate_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\Evaluation\eddy_fluxes\Eddy_Fluxes_Sites_Locations.xlsx'

site_locate_pd = pd.read_excel(site_locate_path)

EC_pos = []

for eddy_flux_name in eddy_flux_name_list:
    eddy_flux_site_name = eddy_flux_name.split('_')[0]
    eddy_flux_site_locate = site_locate_pd[site_locate_pd['Site'] == eddy_flux_site_name]
    eddy_flux_lat = eddy_flux_site_locate['Latitude'].to_numpy()[0]  # x
    eddy_flux_lon = eddy_flux_site_locate['Longitude'].to_numpy()[0]  # y
    EC_pos.append((eddy_flux_lat, eddy_flux_lon))

# In[] Visualization for improved GPP
cmap = plt.cm.RdBu_r
# cmap = plt.cm.PiYG
# norm = mcolors.Normalize(vmin=-1, vmax=500)

light_gray = np.array([0.83, 0.83, 0.83, 1.0])
# colors = cmap(np.linspace(0, 1, cmap.N))
# colors[0] = np.array([0.83, 0.83, 0.83, 1.0])  # Set the first color to gray (for the value -2)
# custom_cmap = mcolors.ListedColormap(colors)

colors = cmap(np.linspace(0, 1, cmap.N))
# colors[:int(cmap.N / 2)] = plt.cm.PuRd_r(np.linspace(0.3, 0.8, cmap.N // 2))
# colors[int(cmap.N / 2):] = plt.cm.Greens(np.linspace(0, 0.8, cmap.N // 2)) #Greens

colors[:int(cmap.N / 2)] = plt.cm.Blues_r(np.linspace(0.3, 0.9, cmap.N // 2))  # 深蓝到浅蓝
colors[int(cmap.N / 2):] = plt.cm.Reds(np.linspace(0.1, 0.7, cmap.N // 2))  # 浅红到深红

custom_cmap = mcolors.ListedColormap(colors)

# norm = mcolors.PowerNorm(gamma=1.0, vmin=-1.0, vmax=1.0)
# norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
# levels = np.linspace(-1, 1, 8)

# levels = np.concatenate((np.linspace(-1, 0, 5), np.linspace(0, 1, 5)[1:]))
# levels = np.linspace(-1.5, 1.5, 9)

levels = [-1.25, -1.00, -0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75, 1.0, 1.25]  # 与图片中的刻度一致
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)

# norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

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
add_shp(ax, lw=0.5, ec='k', fc='none', zorder=3)

ax.imshow(improved_gpp, origin='upper',
          extent=[-79.77497863776252, -44.516103736085974, -20.521922834169924, 8.628408135496414],
          transform=ccrs.PlateCarree(), zorder=2, cmap=custom_cmap, norm=norm, )  # norm=norm,

# plot the EC position
for tmp_i in np.arange(len(eddy_flux_name_list)):
    ax.plot( EC_pos[tmp_i][1],EC_pos[tmp_i][0], marker='*',color='k', #
             alpha=0.9,markersize=12,markeredgecolor='None',zorder=4) #markeredgecolor='None'

    ax.text(EC_pos[tmp_i][1] - 0.6,EC_pos[tmp_i][0]-2, eddy_flux_name_list[tmp_i].split('_')[0],
            color="k", fontsize=14,)  # 半透明黑底白字

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
               labelleft=True, labelright=False, labeltop=True, labelbottom=False, zorder=-1)

# draw a white rectangle around the subplots
left, bottom, width, height = 0.665, 0.145, 0.25, 0.30
ax0 = fig.add_axes([left, bottom, width, height])
plt.Rectangle((left, bottom), width, height, transform=ax0.transAxes, facecolor='white', zorder=2)
# ax0.tick_params(axis='both', which='both', colors='white')
ax0.spines['bottom'].set_color('white')
ax0.spines['top'].set_color('white')
ax0.spines['right'].set_color('white')
ax0.spines['left'].set_color('white')
ax0.tick_params(axis='both', which='both', colors='white', left=False, right=False, top=False, bottom=False,
                labelleft=False, labelright=False, labeltop=False, labelbottom=False, )

left, bottom, width, height = 0.665, 0.23, 0.20, 0.20
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
sns.histplot(data=delat_r2, stat="probability",
             bins=50, color='gray', alpha=0.6, edgecolor='k', linewidth=0.25, ax=ax1, zorder=3)

ax1.yaxis.set_ticks_position('right')
ax1.yaxis.set_label_position('right')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(0, 0.13)
ax1.set_ylabel('Density', fontsize=10)
ax1.set_xlabel('r changes', fontsize=10, labelpad=0.5)

ax1.vlines(0, 0, 0.13, color='darkred', linewidth=1.0, linestyle='--')
ax1.text(0.05, 0.105, '(0<): {}%'.format(round(percent_improved_gpp * 100, 1)), fontsize=8, color='darkred')

# ax1.text(-0.9, 0.04, 'mean r: {:.2}'.format(mean_r2), fontsize=10, color='darkred')
# ax1.text(-0.9, 0.03, 'median r: {:.2}'.format(median_r2), fontsize=10, color='darkred')

# ax1.text(-0.4, 0.04, 'mean r: {:.2}'.format(mean_r2), fontsize=10, color='darkred') # for GLASS EC_LUE
# ax1.text(-0.4, 0.03, 'median r: {:.2}'.format(median_r2), fontsize=10, color='darkred')

# ax1.text(-0.9, 0.055, 'mean r: {:.2}'.format(mean_r2), fontsize=10, color='darkred') # for VPM
# ax1.text(-0.9, 0.04, 'median r: {:.2}'.format(median_r2), fontsize=10, color='darkred')

# Set the background color behind the ticks and labels
ax1.tick_params(axis='both', which='both', colors='black', labelsize=8)

# legend_elements = [Patch(facecolor=light_gray, edgecolor='k', label='Non-Forest')]
# ax.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=False, prop={'size': 12})

# show the colorbar of the deciduousness amplitude
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax, orientation='horizontal',
                    fraction=0.04, pad=0.03)

# 设置刻度标签（完全匹配图片）
cbar.ax.xaxis.set_ticks(np.array([-1.00, -0.50, 0, 0.50, 1.00, ]))
cbar.ax.xaxis.set_ticklabels(['≤ -1.00', '-0.50', '0', '0.50', ' 1.00 ≤ '])

# # 设置刻度线样式
# cbar.ax.xaxis.set_tick_params(
#     length=3,
#     width=1,
#     colors='black',
#     which='both'
# )

cbar.set_label('r changes', fontsize=12)

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Improved_LA_GPP_EC_0521.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# In[] Visualize the rmse of the improved GPP across the Amazon basin
# cmap = plt.cm.RdBu_r
# cmap = plt.cm.PiYG
cmap = plt.cm.BrBG_r

# norm = mcolors.Normalize(vmin=-1, vmax=500)
light_gray = np.array([0.83, 0.83, 0.83, 1.0])

# colors[0] = np.array([0.83, 0.83, 0.83, 1.0])  # Set the first color to gray (for the value -2)
# custom_cmap = mcolors.ListedColormap(colors)
colors = cmap(np.linspace(0, 1, cmap.N))

colors[:int(cmap.N / 2)] = plt.cm.Greens_r(np.linspace(0.2, 0.8, cmap.N // 2))
colors[int(cmap.N / 2):] = plt.cm.Purples(np.linspace(0.2, 0.8, cmap.N // 2))

custom_cmap = mcolors.ListedColormap(colors)

# norm = mcolors.PowerNorm(gamma=1.0, vmin=-1.0, vmax=1.0)
# norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
# levels = np.linspace(-1, 1, 8)

# levels = np.concatenate((np.linspace(-1, 0, 5), np.linspace(0, 1, 5)[1:]))
# levels = np.linspace(-100, 100, 9)
levels = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]  # 与图片中的刻度一致

norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N)

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
add_shp(ax, lw=0.5, ec='k', fc='none', zorder=3)

ax.imshow(rmse, origin='upper',
          extent=[-79.77497863776252, -44.516103736085974, -20.521922834169924, 8.628408135496414],
          transform=ccrs.PlateCarree(), zorder=2, cmap=custom_cmap, norm=norm, )  # norm=norm,

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
               labelleft=True, labelright=False, labeltop=True, labelbottom=False, zorder=-1)

# draw a white rectangle around the subplots
left, bottom, width, height = 0.665, 0.145, 0.25, 0.30
ax0 = fig.add_axes([left, bottom, width, height])
plt.Rectangle((left, bottom), width, height, transform=ax0.transAxes, facecolor='white', zorder=2)
# ax0.tick_params(axis='both', which='both', colors='white')
ax0.tick_params(axis='both', which='both', colors='white', left=False, right=False, top=False, bottom=False,
                labelleft=False, labelright=False, labeltop=False, labelbottom=False, )
ax0.spines['bottom'].set_color('white')
ax0.spines['top'].set_color('white')
ax0.spines['right'].set_color('white')
ax0.spines['left'].set_color('white')

# plt.axis('off')

# ax0.set_facecolor('white')
# ax0.axis('off')

left, bottom, width, height = 0.665, 0.23, 0.20, 0.20
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
sns.histplot(data=rmse_basin, stat="probability",
             bins=100, color='gray', alpha=0.6, edgecolor='k', linewidth=0.25, ax=ax1, zorder=3)

ax1.yaxis.set_ticks_position('right')
ax1.yaxis.set_label_position('right')

ax1.set_xlim(-100, 100)
ax1.set_ylim(0, 0.105)
ax1.set_ylabel('Density', fontsize=10)
ax1.set_xlabel('RMSE changes(%)', fontsize=10, labelpad=0.5)

ax1.vlines(0, 0, 0.105, color='darkred', linewidth=1.0, linestyle='--')
ax1.text(-95, 0.08, '(<0): {}%'.format(round(percent_rmse * 100, 1)), fontsize=8, color='darkred')

# Set the background color behind the ticks and labels
ax1.tick_params(axis='both', which='both', colors='black', labelsize=8)
# ax1.set_ylim(0, 0.2)

# Set axis labels and title with a white background
# ax1.xaxis.label.set_backgroundcolor('white')
# ax1.yaxis.label.set_backgroundcolor('white')


# legend_elements = [Patch(facecolor=light_gray, edgecolor='k', label='Non-Forest')]
# ax.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=False, prop={'size': 12})

# show the colorbar of the deciduousness amplitude
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax, orientation='horizontal',
                    fraction=0.04, pad=0.03)

cbar.set_label('RMSE changes(%)', fontsize=12)

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\RMSE_Amazon_LA_Ensemble_0521.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# In[] rmse reduction of each subregion
from shapely.wkt import loads as load_wkt
import rasterio
import rasterio.mask

shp_file = ogr.Open(
    r"J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\ThreeRegions_Boundary\clip\Amazon_ThreeRegions_Clip.shp")

layer = shp_file.GetLayer()

polygons = [feature.GetGeometryRef().ExportToWkt() for feature in layer]
polygons_name = [feature.GetField('Name') for feature in layer]

# Convert WKT polygons to Shapely geometries
shp_polygon = [load_wkt(polygon) for polygon in polygons]

# In[] Calculate the rmse reduction of each subregion

total_area = len(rmse[~np.isnan(rmse)])

reduce_rmse = [-20.1]
corresponding_gpp = [5.03]

with rasterio.open(rmse_path) as src:
    for i, polygon in enumerate(shp_polygon):
        # Mask the TIFF with the polygon
        #     shapes = [rasterio.features.geometry_mask([polygon], transform=src.transform, invert=True)]
        out_image, out_transform = rasterio.mask.mask(src, [polygon], crop=True)
        out_image = out_image[0]

        out_image = out_image.astype(np.float32)
        out_image[out_image == 65535] = np.nan
        out_image = out_image * 100
        # Remove the first dimension # Calculate the mean value for the region
        mean_value = np.nanmean(out_image[~np.isnan(out_image)])
        std_value = np.nanstd(out_image[~np.isnan(out_image)])

        proportion = out_image[~np.isnan(out_image)].size / total_area

        print('{}, Mean value: {:.1f}, std value: {:.1f}'.format(polygons_name[i], np.round(mean_value, 2),
                                                                 np.round(std_value, 2)))
        print('{}, Proportion: {:.1f}, GPP {:.2f}'.format(polygons_name[i], proportion,
                                                          mean_value * proportion * 25 / 100))

        reduce_rmse.append(round(mean_value, 1))
        corresponding_gpp.append(round(mean_value * proportion * 25 / 100, 1))

# In[] Calculate the r change of each subregion
improved_gpp_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Delta_r2_LA_Ensemble_Two_SIF_basin.tif'

with rasterio.open(improved_gpp_path) as src:
    for i, polygon in enumerate(shp_polygon):
        # Mask the TIFF with the polygon
        #     shapes = [rasterio.features.geometry_mask([polygon], transform=src.transform, invert=True)]
        out_image, out_transform = rasterio.mask.mask(src, [polygon], crop=True)
        out_image = out_image[0]
        out_image = out_image.astype(np.float32)
        out_image = out_image[~np.isnan(out_image)]
        # Remove the first dimension # Calculate the positive proportion for the region
        positive_proportion = np.nansum(out_image > 0) / out_image.size
        print('{}, Positive proportion: {:.3f}'.format(polygons_name[i], positive_proportion))

# In[] Calculate the rmse change of each subregion
rmse_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Delta_rmse_LA_Ensemble_Two_SIF_basin.tif'

with rasterio.open(rmse_path) as src:
    for i, polygon in enumerate(shp_polygon):
        # Mask the TIFF with the polygon
        #     shapes = [rasterio.features.geometry_mask([polygon], transform=src.transform, invert=True)]
        out_image, out_transform = rasterio.mask.mask(src, [polygon], crop=True)
        out_image = out_image[0]
        out_image = out_image.astype(np.float32)
        out_image = out_image[~np.isnan(out_image)]
        # Remove the first dimension # Calculate the positive proportion for the region
        positive_proportion = np.nansum(out_image < 0) / out_image.size
        print('{}, Negative proportion: {:.3f}'.format(polygons_name[i], positive_proportion))

# In[] Visualization for the rmse reduction with its corresponding GPP of each subregion using bar plot


position_x = np.arange(4)

fig = plt.figure(dpi=300, figsize=(9, 6))

left, bottom, width, height = 0.10, 0.15, 0.78, 0.80  # the position of left lower corner, width and height of the figure

ax1 = fig.add_subplot([left, bottom, width, height])

ax1.bar(position_x - 0.4, reduce_rmse, width=0.4, color='#91bfdb', alpha=0.6, edgecolor='k', linewidth=1,
        label='RMSE')

ax2 = ax1.twinx()
ax2.bar(position_x, corresponding_gpp, width=0.4, color='#fc8d59', alpha=0.6, edgecolor='k', linewidth=1,label='GPP')

ax2.set_ylim(0, 2.6)
ax2.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
ax2.set_yticklabels(['0', '0.5', '1', '1.5', '2', '2.5'], fontsize=18)
ax2.set_ylabel('Equivalent GPP (PgC/year)', fontsize=20, labelpad=10)

ax1.set_xticks(position_x - 0.2)
ax1.set_xticklabels(['Amazon Basin', 'Southern Amazon', 'Guiana Shield', 'Everwet Amazon'], fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.set_ylabel('RMSE changes (%)', fontsize=20, labelpad=10)

# figure legend of two bars
legend_elements = [Patch(facecolor='#91bfdb', edgecolor='k', label='RMSE'),
                   Patch(facecolor='#fc8d59', edgecolor='k', label='GPP')]

ax1.legend(handles=legend_elements, loc='upper right', fontsize=16, frameon=False, prop={'size': 16})

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\RMSE_Reduction_GPP_Subregion.png'
# plt.savefig(save_path, dpi=300, bbox_inches='tight')

# In[] Improved GPP and precipitation


# In[] Improved GPP and Asynchrony Degree
asynchrony_range = np.unique(asynchrony)
positive_improved_gpp = delat_r2  # [delat_r2 > 0]
positive_asynchrony = asynchrony  # [delat_r2 > 0]

# asynchrony_range = np.unique(positive_asynchrony)
improve_gpp_list = [np.nanmean(positive_improved_gpp[positive_asynchrony == i]) for i in asynchrony_range]
improve_gpp_std_list = [np.nanstd(positive_improved_gpp[positive_asynchrony == i]) for i in asynchrony_range]
improve_gpp_box = [positive_improved_gpp[positive_asynchrony == i] for i in asynchrony_range]

# In[] density for each box
from matplotlib.colors import Normalize, LinearSegmentedColormap

density_box = [len(improve_gpp_box[i]) / len(positive_improved_gpp) for i in range(len(improve_gpp_box))]

norm = Normalize(vmin=0, vmax=0.6)

normalized_density_box = norm(density_box)
# color_gradient = np.linspace(0.2, 0.8, 10)
feature_color = [plt.cm.Greys(x) for x in normalized_density_box]

# In[] box plot for improved GPP and asynchrony degree
fig = plt.figure(dpi=300)

left, bottom, width, height = 0.15, 0.15, 0.75, 0.78  # the position of left lower corner, width and height of the figure

ax = fig.add_subplot([left, bottom, width, height])

for i in range(len(improve_gpp_box)):
    # ax.boxplot(improve_gpp_box[i], positions=[asynchrony_range[i]], showfliers=False,
    #            showmeans=True, meanline=True, meanprops=dict(color='g', linestyle='-'), medianprops=dict(linewidth=0),
    #            usermedians=None, patch_artist=True, boxprops=dict(facecolor='white', color='g'),
    #            )
    im = sns.boxplot(y=improve_gpp_box[i], positions=[asynchrony_range[i]], ax=ax, linewidth=1.0, showmeans=True,
                     showfliers=False,
                     meanline=True, medianprops=dict(linewidth=0), meanprops=dict(color='k', linestyle='-'),
                     color=feature_color[i])

    print('Asynchrony degree: {}, Mean value: {:.3f}, Std: {:.3f}'.format(asynchrony_range[i], improve_gpp_list[i],
                                                                          improve_gpp_std_list[i]))

ax.set_xlim(-0.5, 6.5)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax.set_xticklabels(['0', '1', '2', '3', '4', '5', '6'], fontsize=14)
ax.set_xlabel('Asynchrony degree (month)', fontsize=16, labelpad=10)

ax.tick_params(axis="y", labelsize=14)

ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(['-0.2', '0', '0.2', '0.4', '0.6', '0.8'], fontsize=14)
ax.set_ylabel('r changes ', fontsize=16, labelpad=10)

sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys, norm=norm)
sm.set_array([])

# cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.03)
# cbar.ax.set_ylim(0, 0.3)
# cbar.set_label('Density', fontsize=16)

# # Create a truncated colormap
cmap = plt.cm.Greys
colors = cmap(np.linspace(0, 0.6, 256))
# Only take the part of the colormap up to 0.6
truncated_cmap = LinearSegmentedColormap.from_list('truncated_greys', colors)
# Add a color bar with the truncated colormap and custom ticks
sm = plt.cm.ScalarMappable(cmap=truncated_cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.03)

# cbar.ax.set_title('Density', fontsize=16, pad=10)
cbar.set_label('Density', fontsize=16)
# Set custom ticks and labels to display the range [0, 0.3]
cbar.set_ticks([0, 0.1, 0.2, 0.3])
cbar.set_ticklabels(['0', '0.1', '0.2', '0.3'], fontsize=14)
# cbar.mappable.set_clim(0, 0.3)
# In[] the relationship between asynchrony degree and improve performance based on the binning method

bins_level = 11

bins_range = np.linspace(-1, 1, bins_level)
#
# bins_range = np.arange(-1, 1.0, 0.04)

# negative_bins = np.linspace(-1, 0, 26)  # 25 bins for the negative range
# positive_bins = np.linspace(0, 1, 26)  # 25 bins for the positive range
#
# bins_range = np.concatenate((negative_bins[:-1], positive_bins))

# bins_range = positive_bins

# improved_gpp_bins = np.digitize(delat_r2, positive_bins)

full_improved_gpp_bins = np.digitize(delat_r2, bins_range)

# im_gpp_mean = [np.nanmean(delat_r2[improved_gpp_bins == i]) for i in range(1, 25+ 1)]
# im_gpp_std = [np.nanstd(delat_r2[improved_gpp_bins == i]) for i in range(1, 25 + 1)]

# asyn_mean = [np.nanmean(asynchrony[improved_gpp_bins == i]) for i in range(1, 25 + 1)]
# asyn_std = [np.nanstd(asynchrony[improved_gpp_bins == i]) for i in range(1, 25 + 1)]


im_gpp_mean = [np.nanmean(delat_r2[full_improved_gpp_bins == i]) for i in range(1, bins_level + 1)]
im_gpp_std = [np.nanstd(delat_r2[full_improved_gpp_bins == i]) for i in range(1, bins_level + 1)]

asyn_mean = [np.nanmean(asynchrony[full_improved_gpp_bins == i]) for i in range(1, bins_level + 1)]
asyn_std = [np.nanstd(asynchrony[full_improved_gpp_bins == i]) for i in range(1, bins_level + 1)]

density_improved_gpp = [len(full_improved_gpp_bins[full_improved_gpp_bins == i]) / len(full_improved_gpp_bins) for i in
                        range(1, bins_level + 1)]

asyn_box = [asynchrony[full_improved_gpp_bins == i] for i in range(1, bins_level + 1)]

# img_gpp_box = [delat_r2[asynchrony == i] for i in range(0, 7)]

# plt.bar([x+1/(len(bins_range) - 1) for x in range(len(bins_range))], density_improved_gpp)
#
# plt.xticks(range(0, len(bins_range), 2), [round(bins_range[x], 1) for x in range(0, len(bins_range), 2)])
# plt.hist(asynchrony)

test_asyn = asynchrony[(delat_r2 < 0.5) & (delat_r2 > 0.0)]

plt.boxplot(test_asyn, showmeans=True)
for i in range(len(asyn_box)):
    plt.boxplot(asyn_box[i])

# In[]Create a figure and axis for box plot with asynchrony degree and improved performance

data = asyn_box

# Generate sample data

# Calculate densities
densities = [len(d) for d in data]
max_density = max(densities)
colors = plt.cm.viridis([d / max_density for d in densities])

# alphas = [d / max_density for d in densities] # Normalize densities to use as alphas
min_alpha = 0.01
alphas = [max(d / max_density, min_alpha) for d in densities]  # Normalize densities and apply minimum alpha
fig, ax = plt.subplots(figsize=(12, 8))

# Custom box plot with standard deviation boundaries
for i, sample in enumerate(data):
    mean = np.mean(sample)
    std_dev = np.std(sample)
    count = len(sample)

    # Calculate the positions of the box plot components
    lower_bound = mean - 0.25 * std_dev
    upper_bound = mean + 0.25 * std_dev

    # Draw box
    ax.add_patch(plt.Rectangle((i - 0.15, lower_bound), 0.3, 0.5 * std_dev, color=colors[i], alpha=0.5))
    # Plot median line
    ax.plot([i - 0.15, i + 0.15], [mean, mean], color='darkred', linewidth=1.5, zorder=10)

    # Plot whiskers
    lower_bound = mean - std_dev
    upper_bound = mean + std_dev
    ax.plot([i, i], [mean - 0.25 * std_dev, lower_bound], color='#2b83ba', linewidth=1, alpha=0.7, zorder=4)
    ax.plot([i, i], [mean + 0.25 * std_dev, upper_bound], color='#2b83ba', linewidth=1, alpha=0.7, zorder=4)

    # Plot outliers
    # outliers = sample[(sample < lower_bound) | (sample > upper_bound)]
    # ax.scatter([i]*len(outliers), outliers, color='red', zorder=10)

# Set labels and title
ax.set_xticks(range(len(data)))
# ax.set_xticklabels([f'Sample {i+1}' for i in range(len(data))])
# ax.set_title('Custom Box Plot with Standard Deviation Boundaries and Data Density Colors')
# ax.set_ylabel('Values')

plt.show()

# plt.boxplot(asyn_box)

# ax2.boxplot(evergreen_bins_list, positions=bins + 2.4, widths=4, showfliers=False, patch_artist=True,
#             boxprops=dict(facecolor='white', color='g', ),
#             whiskerprops=dict(color='g'), capprops=dict(color='g'),
#             medianprops=dict(color='k'))  #,showmeans=True,meanline = True, meanprops=dict(color='g', linestyle='--')


# stat_p = [stats.pearsonr(delat_r2[improved_gpp_bins == i], asynchrony[improved_gpp_bins == i]) for i in range(1, bins_level + 1)]

# t_test_list = [stats.ttest_ind(delat_r2[improved_gpp_bins == i], asynchrony[improved_gpp_bins == i]) for i in
#                range(1, bins_level + 1)]

# In[] box with density

r = round(stats.pearsonr(im_gpp_mean, asyn_mean)[0], 2)
p = round(stats.pearsonr(im_gpp_mean, asyn_mean)[1], 2)

fig, ax = plt.subplots(1, 1, dpi=300, figsize=(6.5, 5))

# scatter trend
# ax.scatter(im_gpp_mean, asyn_mean, s=20, alpha=0.5, color='#2b83ba')
# ax.fill_between(im_gpp_mean, np.array(asyn_mean) - np.array(asyn_std), np.array(asyn_mean) + np.array(asyn_std),
#                 alpha=0.2, color='#2b83ba')

for i, sample in enumerate(asyn_box[:-1]):
    ind = bins_range[i]
    mean = np.mean(sample)
    std_dev = np.std(sample)
    count = len(sample)

    # Calculate the positions of the box plot components
    lower_bound = mean - 0.25 * std_dev
    upper_bound = mean + 0.25 * std_dev

    # Draw box
    ax.add_patch(plt.Rectangle((ind - 0.15, lower_bound), 0.3, 0.5 * std_dev, color='#2b83ba', alpha=0.5))
    # Plot median line
    ax.plot([ind - 0.15, ind + 0.15], [mean, mean], color='darkred', linewidth=1.5, zorder=10)

    # Plot whiskers
    lower_bound = mean - std_dev
    upper_bound = mean + std_dev
    ax.plot([ind, ind], [mean - 0.25 * std_dev, lower_bound], color='#2b83ba', linewidth=1, alpha=0.7, zorder=4)
    ax.plot([ind, ind], [mean + 0.25 * std_dev, upper_bound], color='#2b83ba', linewidth=1, alpha=0.7, zorder=4)

ax.set_ylim(-1, 6)
ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6'], fontsize=14)

ax.set_xlim(-1.04, 1.04)
ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xticklabels(['-1', '-0.8', '-0.6', '-0.4', '-0.2', '0', '0.2', '0.4', '0.6', '0.8', '1'], fontsize=14)

ax.set_ylabel('Asynchrony Degree (m)', fontsize=16)

ax.yaxis.set_label_coords(-0.075, 0.5)

ax.set_xlabel('Improved Performance', fontsize=16)
# ax.tick_params(axis='both', which='major', labelsize=14)

ax.text(-0.9, 5.5, 'R = ' + str(r), fontsize=14)
ax.text(-0.9, 5.0, 'p < 0.001', fontsize=14)

ax2 = ax.twinx()

# plot the density of improved GPP
# ax2.plot(bins_range, density_improved_gpp, label='Improved GPP Density')
ax2.bar(bins_range[:-1] + 1 / (len(bins_range) - 1), density_improved_gpp[:-1], width=2 / (len(bins_range) - 1),
        alpha=0.6,
        label='Improved GPP Density',
        edgecolor='k', linewidth=0.5, color='#bababa')  # [:-1]

ax2.vlines(0, 0, 0.2, color='darkred', linewidth=1.0, linestyle='--')
ax2.text(0.05, 0.16, '(>0): {}%'.format(round(percent_improved_gpp * 100, 2)), fontsize=14, color='darkred')

ax2.set_ylim(0, 0.4)
ax2.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20])
ax2.set_yticklabels(['0.00', '0.05', '0.10', '0.15', '0.20'], fontsize=14)
ax2.set_ylabel('Density', fontsize=16)
# ax2.yaxis.label.set(position=(0.0, 0.2))
ax2.yaxis.set_label_coords(1.125, 0.2)

plt.tight_layout()

# In[]

r = round(stats.pearsonr(im_gpp_mean, asyn_mean)[0], 2)
p = round(stats.pearsonr(im_gpp_mean, asyn_mean)[1], 2)

fig, ax = plt.subplots(1, 1, dpi=300, figsize=(6.5, 5))

# scatter trend
ax.scatter(im_gpp_mean, asyn_mean, s=20, alpha=0.5, color='#2b83ba')
ax.fill_between(im_gpp_mean, np.array(asyn_mean) - np.array(asyn_std), np.array(asyn_mean) + np.array(asyn_std),
                alpha=0.2, color='#2b83ba')

ax.set_ylim(-1, 6)
ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6'], fontsize=14)

ax.set_xlim(-1.04, 1.04)
ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xticklabels(['-1', '-0.8', '-0.6', '-0.4', '-0.2', '0', '0.2', '0.4', '0.6', '0.8', '1'], fontsize=14)

ax.set_ylabel('Asynchrony Degree (m)', fontsize=16)

ax.yaxis.set_label_coords(-0.075, 0.5)

ax.set_xlabel('Improved Performance', fontsize=16)
# ax.tick_params(axis='both', which='major', labelsize=14)

ax.text(-0.9, 5.5, 'R = ' + str(r), fontsize=14)
ax.text(-0.9, 5.0, 'p < 0.001', fontsize=14)

ax2 = ax.twinx()

# plot the density of improved GPP
# ax2.plot(bins_range, density_improved_gpp, label='Improved GPP Density')
ax2.bar(bins_range[:-1] + 1 / (len(bins_range) - 1), density_improved_gpp[:-1], width=2 / (len(bins_range) - 1),
        alpha=0.6,
        label='Improved GPP Density',
        edgecolor='k', linewidth=0.5, color='#bababa')  # [:-1]

ax2.vlines(0, 0, 0.2, color='darkred', linewidth=1.0, linestyle='--')
ax2.text(0.05, 0.16, '(>0): {}%'.format(round(percent_improved_gpp * 100, 2)), fontsize=14, color='darkred')

ax2.set_ylim(0, 0.4)
ax2.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20])
ax2.set_yticklabels(['0.00', '0.05', '0.10', '0.15', '0.20'], fontsize=14)
ax2.set_ylabel('Density', fontsize=16)
# ax2.yaxis.label.set(position=(0.0, 0.2))
ax2.yaxis.set_label_coords(1.125, 0.2)

plt.tight_layout()

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Improved_GPP_Asynchrony.png'
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
# print('r:', round(stats.pearsonr(im_gpp_mean, asyn_mean)[0], 2))
# print('p:', round(stats.pearsonr(im_gpp_mean, asyn_mean)[1], 2))

# In[] distribution of improved GPP across the climate zones in Amazon basin

par_path = r'Z:\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\climate_par_mean.tif'
rainfall_path = r'Z:\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\climate_precipitation_sum.tif'
# rainfall_path = r'Z:\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\Visualization_Variables\soil_fertility_visualized.tif'

_geo, _prj, par = readTif_gdal(par_path)
_geo, _prj, rainfall = readTif_gdal(rainfall_path)

par = cv2.resize(par, (raws_y, columns_x))
rainfall = cv2.resize(rainfall, (raws_y, columns_x))

rainfall[np.isinf(rainfall)] = np.nan

ampl_dec_res_forest = improved_gpp[~np.isnan(improved_gpp)]

# ampl_dec_res_forest = ampl_dec[~np.isnan(ampl_dec)]
# par_res = par[~np.isnan(ampl_dec)]
# rainfall_res = rainfall[~np.isnan(ampl_dec)]
# asynchrony_res = time_lag[~np.isnan(ampl_dec)]

# mask = ampl_dec_res_forest < 1000
par_res = par[~np.isnan(improved_gpp)]
rainfall_res = rainfall[~np.isnan(improved_gpp)]
# ampl_dec_res_forest = ampl_dec_res_forest / 10

# In[] bin relationship between ampl dec and asynchrony

bins = np.arange(0, 50, 5)
dec_bins = np.digitize(ampl_dec_res_forest, bins)

# dec_statistics = [len(ampl_dec_res_forest[dec_bins == i]) / len(ampl_dec_res_forest) for i in range(1, len(bins) + 1)]

dec_statistics = [np.nanmean(ampl_dec_res_forest[dec_bins == i]) for i in range(1, len(bins) + 1)]
asyn_mean = [np.nanmean(asynchrony_res[dec_bins == i]) for i in range(1, len(bins) + 1)]

rain_bin_mean = [np.nanmean(rainfall_res[dec_bins == i]) for i in range(1, len(bins) + 1)]

fig, ax = plt.subplots(1, 1, dpi=300, figsize=(6.5, 5))

ax.scatter(dec_statistics, asyn_mean, s=20, alpha=0.5, color='#2b83ba')
ax2 = ax.twinx()
ax2.bar(dec_statistics, rain_bin_mean, width=4, alpha=0.6, label='Rainfall', edgecolor='k', linewidth=0.5,
        color='#bababa')
ax2.set_ylim(1500, 3500)

# In[]


# Bin the data frame by "rainfall-bins" with a bin size of 10
# bins = np.linspace(0, 100, 10000) # for dec amplitude
bins_rain = np.linspace(np.nanmin(rainfall_res), np.nanmax(rainfall_res), 50)
bins_par = np.linspace(np.nanmin(par_res), np.nanmax(par_res), 50)

rainfall_bins = np.digitize(rainfall_res, bins_rain)
par_bins = np.digitize(par_res, bins_par)

rain_par_dec = np.zeros((len(bins_rain), len(bins_par))) * np.nan

for i in range(1, len(bins_rain) + 1):
    dec_for_rain_bin = ampl_dec_res_forest[rainfall_bins == i]
    par_bins_for_rain_bin = par_bins[rainfall_bins == i]
    for j in range(1, len(bins_par) + 1):
        rain_par_dec[i - 1, j - 1] = np.nanmean(dec_for_rain_bin[(par_bins_for_rain_bin == j)])

# In[]

plt.figure()
plt.pcolormesh(bins_rain, bins_par / 10, rain_par_dec, cmap=custom_cmap, norm=norm)
# plt.pcolormesh(bins_rain, bins_par / 10, rain_par_dec,  cmap='RdBu_r', vmin=-1, vmax=1)

plt.colorbar()
plt.xlabel('MAP (mm/yr)', fontsize=16)
# plt.xlabel('Soil Fertility', fontsize=16)
plt.ylabel('Par (W/m$^2$)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.xlim(350, 5000)
plt.ylim(115, 225)

plt.tight_layout()

# In[]
# plt.figure(dpi=300, figsize=(6, 4))
#
# plt.bar(np.arange(50), density_improved_gpp, label='Improved GPP Density')
# plt.figure(dpi=300, figsize=(6, 4))
#
# sns.histplot(data=delat_r2, stat='probability', kde=False, bins=50, alpha=0.5, )
# # # plt.axvline(x=0, color='red', linewidth=1.5)
# # # plt.text(10.6, 0.24, 'x = 10%', fontsize=14, color='r')
# # # plt.hist(data=delat_r2, stat='probability', kde=True, bins=20, alpha=0.6)
# # plt.xlim(-1.0, 1.0)
# # plt.xlabel('Improved Performance', fontsize=16)
# # plt.ylabel('Density', fontsize=16)
# #
# # plt.xticks(fontsize=14)
# # plt.yticks(fontsize=14)
# # plt.tight_layout()
# #
# # # np.nansum(delat_r2 > 0) / len(delat_r2)
# print(np.nansum(delat_r2[~np.isnan(delat_r2)] > 0) / np.nansum(~np.isnan(delat_r2)))

# In[]
# plt.hexbin(delat_r2, asynchrony, gridsize=50, cmap='Blues', bins='log')
