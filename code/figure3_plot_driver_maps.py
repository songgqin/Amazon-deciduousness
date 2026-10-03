"""Figure 3 driver map plotting.

"""

# In[] Imports
import copy

# In[] Workflow
baseAtla2 = {}
baseAtla2[1] = [[233, 233, 233], [201, 90, 90], [100, 173, 191], [86, 64, 71]]
baseAtla2[2] = [[249, 244, 248], [139, 182, 224], [235, 130, 144], [122, 123, 171]]
baseAtla2[3] = [[243, 243, 243], [140, 227, 176], [231, 164, 209], [124, 143, 176]]
baseAtla2[4] = [[233, 233, 233], [201, 180, 90], [154, 115, 176], [129, 76, 51]]
baseAtla2[5] = [[222, 222, 222], [0, 110, 175], [205, 0, 31], [74, 33, 76]]
baseAtla2[6] = [[233, 233, 233], [108, 132, 182], [116, 175, 129], [37, 90, 91]]
baseAtla2[7] = [[233, 233, 233], [89, 201, 201], [191, 100, 173], [56, 71, 149]]
baseAtla2[8] = [[243, 243, 243], [79, 158, 195], [243, 180, 0], [40, 40, 40]]
baseAtla2[9] = [[233, 231, 242], [78, 174, 209], [223, 78, 167], [37, 19, 139]]
baseAtla2[10] = [[227, 228, 223], [127, 135, 124], [57, 120, 164], [20, 72, 83]]
baseAtla2[11] = [[243, 232, 156], [248, 161, 127], [207, 102, 148], [92, 82, 166]]
baseAtla2[12] = [[248, 225, 152], [181, 167, 124], [108, 119, 149], [65, 92, 170]]

baseAtla3 = {}
baseAtla3[1] = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
baseAtla3[2] = [[0, 211, 166], [211, 152, 255], [233, 164, 67]]
baseAtla3[3] = [[249, 3, 252], [252, 242, 8], [2, 251, 249]]
baseAtla3[4] = [[215, 0, 102], [0, 137, 52], [3, 79, 162]]
baseAtla3[5] = [[243, 180, 0], [180, 102, 0], [0, 0, 0]]
baseAtla3[6] = [[31, 5, 94], [128, 128, 128], [180, 0, 0]]
baseAtla3[7] = [[1.0 * 255, 0.4980392156862745 * 255, 0.054901960784313725 * 255],
                [0.12156862745098039 * 255, 0.4666666666666667 * 255, 0.7058823529411765 * 255],
                [0.8392156862745098 * 255, 0.15294117647058825 * 255, 0.1568627450980392 * 255]]

import numpy as np
import itertools

# In[] Functions
def getCMap2(A, B, atla, n):
    nA = (A.flatten() - np.min(A)) / (np.max(A) - np.min(A))
    nB = (B.flatten() - np.min(B)) / (np.max(B) - np.min(B))
    nA = (np.ceil(nA * n) - 1) / (n - 1)
    nB = (np.ceil(nB * n) - 1) / (n - 1)
    nA[nA < 0] = 0
    nB[nB < 0] = 0

    colorAB = ((nA * nB).reshape((-1, 1))) * (atla[3, :].reshape((1, -1))) + \
              ((nA * (1 - nB)).reshape((-1, 1))) * (atla[1, :].reshape((1, -1))) + \
              (((1 - nA) * nB).reshape((-1, 1))) * (atla[2, :].reshape((1, -1))) + \
              (((1 - nA) * (1 - nB)).reshape((-1, 1))) * (atla[0, :].reshape((1, -1)))

    CMap = np.zeros((A.shape[0], A.shape[1], 3))
    CMap[:, :, 0] = np.reshape(colorAB[:, 0], A.shape)
    CMap[:, :, 1] = np.reshape(colorAB[:, 1], A.shape)
    CMap[:, :, 2] = np.reshape(colorAB[:, 2], A.shape)

    return CMap

def getLegend2(atla, n):

    AA, BB = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    XX, YY = np.meshgrid(np.linspace(-1, 1, n + 1), np.linspace(-1, 1, n + 1))

    tCMap = getCMap2(AA, BB, atla, n)

    rectangles = []
    colorWeights = []
    for j in range(n):
        for i in range(n):
            rectangles.append([(XX[i, j], YY[i, j]), (XX[i + 1, j], YY[i + 1, j]), (XX[i + 1, j + 1], YY[i + 1, j + 1]),
                               (XX[i, j + 1], YY[i, j + 1])])
            colorWeights.append([tCMap[i, j, 0], tCMap[i, j, 1], tCMap[i, j, 2]])

    return rectangles, colorWeights

def getCMap3(A, B, C, atla, n):
    nA = (A.flatten() - np.min(A)) / (np.max(A) - np.min(A))
    nB = (B.flatten() - np.min(B)) / (np.max(B) - np.min(B))
    nC = (C.flatten() - np.min(C)) / (np.max(C) - np.min(C))

    nA = (np.ceil(nA * n * 3) - 1) / (n * 3 - 1)
    nB = (np.ceil(nB * n * 3) - 1) / (n * 3 - 1)
    nC = (np.ceil(nC * n * 3) - 1) / (n * 3 - 1)
    nA[nA < 0] = 0
    nB[nB < 0] = 0
    nC[nC < 0] = 0

    ABC = np.column_stack((nA, nB, nC))

    colorABC = (atla[0, :].reshape((1, -1))) * (ABC[:, 0].reshape((-1, 1))) + \
               (atla[1, :].reshape((1, -1))) * (ABC[:, 1].reshape((-1, 1))) + \
               (atla[2, :].reshape((1, -1))) * (ABC[:, 2].reshape((-1, 1)))

    colorABC[np.isnan(colorABC)] = 1
    CMap = np.zeros((A.shape[0], A.shape[1], 3))
    CMap[:, :, 0] = np.reshape(colorABC[:, 0], A.shape)
    CMap[:, :, 1] = np.reshape(colorABC[:, 1], A.shape)
    CMap[:, :, 2] = np.reshape(colorABC[:, 2], A.shape)

    return CMap

def getLegend3(atla, n):

    p1, p2, p3 = (1 / 2, np.sqrt(3) / 2), (0, 0), (1, 0)
    YList = np.linspace(p1[1], p2[1], n + 1)
    XList_L = np.linspace(p1[0], p2[0], n + 1)
    XList_R = np.linspace(p1[0], p3[0], n + 1)

    triangles = []
    colorWeights = []
    for i in range(n):
        XList_i = np.linspace(XList_L[i], XList_R[i], i + 1)
        XList_i_next = np.linspace(XList_L[i + 1], XList_R[i + 1], i + 1 + 1)
        YList_i = np.repeat(YList[i], i + 1)
        YList_i_next = np.repeat(YList[i + 1], i + 1 + 1)

        pts1 = list(zip(XList_i, YList_i))
        pts2 = list(zip(XList_i_next, YList_i_next))
        pts = sorted(pts1 + pts2, key=lambda x: x[0])

        colorRs = [n - (i + 1)] * (2 * (i + 1) - 1)
        colorGs = [[i]] + [[w, w] for w in np.arange(i)[::-1]]
        colorGs = list(itertools.chain.from_iterable(colorGs))
        colorBs = [[w, w] for w in np.arange(i)] + [[i]]
        colorBs = list(itertools.chain.from_iterable(colorBs))

        for k in range(2 * (i + 1) - 1):
            triangles.append([pts[k], pts[k + 1], pts[k + 2]])
            colorWeights.append([colorRs[k] / (n - 1), colorGs[k] / (n - 1), colorBs[k] / (n - 1)])
    AA = np.array(colorWeights)[:, 0].reshape(-1, 1)
    BB = np.array(colorWeights)[:, 1].reshape(-1, 1)
    CC = np.array(colorWeights)[:, 2].reshape(-1, 1)
    tCMap = getCMap3(AA, BB, CC, atla, n)
    colorWeights = tCMap[:, 0, :]
    return triangles, colorWeights

import cartopy.crs as ccrs
from osgeo import gdal
import numpy as np
from scipy import stats
from matplotlib.gridspec import GridSpec
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cmaps
from amazon_preprocessing import readTif_gdal
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import copy

import numpy.ma as ma

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams["axes.unicode_minus"] = True
matplotlib.use("Agg")
matplotlib.rcParams['figure.dpi'] = 150
cm2in = 1 / 2.54

def add_shp(ax, **kwargs):
    proj = ccrs.PlateCarree()

    reader = shpreader.Reader(r'data/boundaries/Amazon_ThreeRegions_Clip.shp')

    provinces = reader.geometries()
    ax.add_geometries(provinces, proj, **kwargs)
    reader.close()

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

data_path =  r'data/drivers/asynchrony_driver_map_3type.tif'

im_geotrans, _proj, drivers = readTif_gdal(data_path)

im_width = drivers.shape[1]
im_height = drivers.shape[0]

lon = np.linspace(im_geotrans[0], im_geotrans[0] + im_geotrans[1] * (im_width - 1), im_width)
lat = np.linspace(im_geotrans[3] + im_geotrans[5] * (im_height - 1), im_geotrans[3], im_height)
print(lon.min(), lon.max(), lat.min(), lat.max())

data_r, data_g, data_b = drivers[:, :, 0], drivers[:, :, 1], drivers[:, :, 2]

cond = np.isnan(data_r) | np.isnan(data_g) | np.isnan(data_b)
data_r = ma.masked_where(cond, data_r)
data_g = ma.masked_where(cond, data_g)
data_b = ma.masked_where(cond, data_b)

fig, ax = plt.subplots(1, 3, figsize=(12, 5), dpi=300)
plt.subplots_adjust(wspace=0.3)
ax[0].imshow(data_r, cmap='jet')
ax[0].set_title('Light')
cax = fig.add_axes([ax[0].get_position().x1 + 0.01, ax[0].get_position().y0, 0.02, ax[0].get_position().height])
fig.colorbar(ax[0].imshow(data_r, cmap='jet'), cax=cax)
ax[0].set_axis_off()

ax[1].imshow(data_g, cmap='jet', vmax=1.0)
ax[1].set_title('Hydroclimate')
cax1 = fig.add_axes([ax[1].get_position().x1 + 0.01, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
fig.colorbar(ax[1].imshow(data_g, cmap='jet', vmax=1.0), cax=cax1)
ax[1].set_axis_off()

ax[2].imshow(data_b, cmap='jet')
ax[2].set_title('Soil')
cax2 = fig.add_axes([ax[2].get_position().x1 + 0.01, ax[2].get_position().y0, 0.02, ax[2].get_position().height])
fig.colorbar(ax[2].imshow(data_b, cmap='jet'), cax=cax2)
ax[2].set_axis_off()

n = 20

baseAtla3[8] = [[208, 28, 139], [253, 184, 99], [65, 182, 196]]

color_list = [
    'tab:orange',
    'tab:brown',
    'lightgray',
    'tab:blue',
]

baseAtla3[9] = [[208, 28, 139], [65, 182, 196],[253, 184, 99] ]

atla3 = np.array(baseAtla3[9]) / 255

triangles, colorWeights3 = getLegend3(atla3, n)

tCMap3 = getCMap3(data_r, data_g, data_b, atla3, n)

cond1 = ~cond
expanded_cond = cond1[:, :, np.newaxis]
masked_tCMap3 = np.where(expanded_cond, tCMap3, 1)

plt.imshow(masked_tCMap3)

gamma = 1.2
data_gamma_corrected = np.power(masked_tCMap3, gamma)

plt.imshow(data_gamma_corrected)

feature_name = ['Light', 'Hydroclimate', 'Soil' ]

dominant = copy.deepcopy(drivers)
mask_basin = np.nansum(dominant, axis=2) > 0
dominant_res = dominant[mask_basin,:]
counts_pro = np.nanmean(dominant_res, axis=0)

color_list = copy.deepcopy(np.array(baseAtla3[9])/255)

re_rank = np.argsort(-counts_pro)
re_rank_name = [feature_name[i] for i in re_rank]
re_rank_color = [color_list[i] for i in re_rank]
re_rank_counts = [counts_pro[i] for i in re_rank]

print('Proportion of the Basin Hydroclimate: {:.3f}, PAR: {:.3f}, Soil: {:.3f}'.format(counts_pro[1], counts_pro[0], counts_pro[2]))

fig = plt.figure(dpi=300)

left, bottom, width, height = 0, 0.01, 0.98, 0.93
proj = ccrs.PlateCarree()
ax = fig.add_subplot([left, bottom, width, height], projection=proj)

ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
ax.add_feature(cfeature.OCEAN, zorder=2)

add_shp(ax, lw=0.6, ec='k', fc='none', zorder=3)

im = ax.imshow(data_gamma_corrected, origin='upper',
               extent=[-79.77497863776252, -44.516103736085974, -20.521922834169924, 8.628408135496414],
               transform=ccrs.PlateCarree(), zorder=1)

extents = [-80, -44, -25, 10]

ax.set_extent(extents, crs=proj)

ax.set_xticks(np.arange(-80, -40, 10), crs=proj)
ax.set_yticks(np.arange(-20, 10 + 10, 10), crs=proj)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.tick_params(axis='both', labelsize=12, direction='out', colors='k', length=3, width=0.9, which='major',
               left=True, right=False, top=True, bottom=False,
               labelleft=True, labelright=False, labeltop=True, labelbottom=False, zorder=2)

left, bottom, width, height = 0.61, 0.05, 0.25, 0.25

ax0 = fig.add_axes([left, bottom, width, height])
rect=plt.Rectangle((left, bottom), width, height, transform=ax0.transAxes, facecolor='white', zorder=10)
ax0.add_patch(rect)
ax0.tick_params(axis='both', which='both', colors='white', left=False, right=False, top=False, bottom=False,
                labelleft=False, labelright=False, labeltop=False, labelbottom=False, )

ax0.spines['bottom'].set_color('white')
ax0.spines['top'].set_color('white')
ax0.spines['right'].set_color('white')
ax0.spines['left'].set_color('white')

left, bottom, width, height = 0.63, 0.090, 0.19, 0.19
ax2 = fig.add_axes([left, bottom, width, height])

ax2.bar(re_rank_name, re_rank_counts, color=re_rank_color, edgecolor='black')
ax2.set_ylim(0, 0.65)
ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position('right')
ax2.set_ylabel('Proportion of the area', fontsize=10)
ax2.set_xlabel('Drivers', fontsize=10)
ax2.set_xticklabels(re_rank_name, ha='center')
ax2.tick_params(axis='both', which='both', colors='k', left=False, right=True, top=False, bottom=True,
                labelleft=False, labelright=True, labeltop=False, labelbottom=True, labelsize=7)
for p in ax2.patches:

    x = p.get_x() + p.get_width() / 2
    y = p.get_height() + 0.02

    ax2.text(x, y, f'{p.get_height() * 100:.1f}' + "%", ha='center',
            va='bottom',
            fontsize=6,
            color='black')

patches = [mpatches.Patch(color=color_list[i], label="{l}".format(l=feature_name[i])) for i in range(len(feature_name))]

left, bottom, width, height = 0.20, 0.045, 0.21, 0.20
ax1 = fig.add_axes([left, bottom, width, height])

polygons = [Polygon(triangle, closed=True) for triangle in triangles]

collection = PatchCollection(polygons, facecolors=colorWeights3, edgecolors=colorWeights3)

ax1.add_collection(collection)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

ax1.axis('off')

ax1.set_facecolor('none')

coords = [
    (1 / 2, np.sqrt(3) / 2+0.075),
    (0.0, -0.13),
    (1, -0.13)
]

labels = ['Light', 'Hydroclimate', 'Soil']

for coord, label in zip(coords, labels):
    ax1.text(
        coord[0],
        coord[1],
        label,
        fontsize=10,
        fontname='Arial',
        horizontalalignment='center'
    )

arrows = [
    (0.25, 0.5 * np.sqrt(3) / 2, -120),
    (0.5, -0.1, 0),
    (0.75, 0.5 * np.sqrt(3) / 2, 120)
]

import os
os.makedirs(r'outputs/figures', exist_ok=True)
fig.savefig(r'outputs/figures/Fig3_Driver_Map.png', dpi=300, bbox_inches='tight')
plt.show()

color_list = [
    'tab:orange',
    'tab:blue',
    'lightgray',

    'tab:brown',
]
custom_cmap = mcolors.ListedColormap(color_list)

feature_name = ['Light', 'Hydroclimate', 'Soil', 'Herbivory', ]

fig = plt.figure(dpi=300)

left, bottom, width, height = 0, 0.08, 0.98, 0.84

proj = ccrs.PlateCarree()
ax = fig.add_subplot([left, bottom, width, height], projection=proj)

ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
ax.add_feature(cfeature.OCEAN, zorder=2)

add_shp(ax, lw=0.5, ec='k', fc='none', zorder=3)

plt.imshow(dominant, origin='upper', cmap=custom_cmap,
           extent=[-79.77497863776252, -44.516103736085974, -20.521922834169924, 8.628408135496414],
           transform=ccrs.PlateCarree(), zorder=2)

patches = [mpatches.Patch(color=color_list[i], label="{l}".format(l=feature_name[i])) for i in range(len(feature_name))]

plt.legend(handles=patches, bbox_to_anchor=(0.99, 0.20), borderaxespad=0.05, facecolor='white', framealpha=1,
           fontsize=8)

extents = [-80, -44, -22, 10]
ax.set_extent(extents, crs=proj)

ax.set_xticks(np.arange(-80, -40, 10), crs=proj)
ax.set_yticks(np.arange(-20, 10 + 10, 10), crs=proj)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.tick_params(axis='both', labelsize=10, direction='out', colors='k', length=3, width=0.9, which='major',
               left=True, right=False, top=True, bottom=False,
               labelleft=True, labelright=False, labeltop=True, labelbottom=False, zorder=10)

save_path = r'outputs/figures/Dominant_asynchrony_herbivory_four_class.png'
fig.savefig(save_path, dpi=300, bbox_inches='tight')

from shapely.wkt import loads as load_wkt
import rasterio
import rasterio.mask
from osgeo import ogr

shp_path = r'data/boundaries/Amazon_ThreeRegions_Clip.shp'

shp_file = ogr.Open(shp_path)

layer = shp_file.GetLayer()

polygons = [feature.GetGeometryRef().ExportToWkt() for feature in layer]
polygons_name = [feature.GetField('Name') for feature in layer]

shp_polygon = [load_wkt(polygon) for polygon in polygons]

feature_name = ['PAR', 'Hydroclimate', 'Soil', 'Vegetation', 'VPD', ]
feature_name = ['PAR', 'Hydroclimate', 'Soil']

dominant_path = r'data/drivers/dominant_driver_map.tif'

reduce_rmse = [8.4]
corresponding_gpp = [2.1]

with rasterio.open(dominant_path) as src:
    for i, polygon in enumerate(shp_polygon):

        out_image, out_transform = rasterio.mask.mask(src, [polygon], crop=True)
        out_image = out_image[0]
        out_image = out_image.astype(np.float32)
        out_image_res = out_image[~np.isnan(out_image)]

        unique, counts = np.unique(out_image_res, return_counts=True)
        counts_pro = counts / np.sum(counts)

        print(polygons_name[i], 'Herbivory classes:', np.round(counts_pro, 3))

dominant_res = dominant[~np.isnan(dominant)]
unique, counts = np.unique(dominant_res, return_counts=True)
counts_pro = counts / np.sum(counts)
print('Proportion of the area by herbivory class:', np.round(counts_pro, 3))

dominant_path = r'data/drivers/asynchrony_driver_map_3type.tif'

with rasterio.open(dominant_path) as src:
    for i, polygon in enumerate(shp_polygon):

        out_image, out_transform = rasterio.mask.mask(src, [polygon], crop=True)

        out_image = out_image.transpose(1, 2, 0)
        out_image = out_image.astype(np.float32)
        mask_img = np.nansum(out_image, 2)>0
        out_image_res = out_image[mask_img,:]

        counts_pro = np.nanmean(out_image_res, axis=0)

        print(polygons_name[i],'Hydroclimate: {:.3f}, PAR: {:.3f}, Soil: {:.3f}'.format(counts_pro[1], counts_pro[0], counts_pro[2]))

_geo, _proj, dominant = readTif_gdal(dominant_path)
mask_basin = np.nansum(dominant, axis=2) > 0
dominant_res = dominant[mask_basin,:]
counts_pro = np.nanmean(dominant_res, axis=0)
print('Proportion of the Basin Hydroclimate: {:.3f}, PAR: {:.3f}, Soil: {:.3f}'.format(counts_pro[1], counts_pro[0], counts_pro[2]))

dominant_res = dominant[~np.isnan(dominant)]
unique, counts = np.unique(dominant_res, return_counts=True)
counts_pro = counts / np.sum(counts)
print('Proportion of the area by driver class:', np.round(counts_pro, 3))
