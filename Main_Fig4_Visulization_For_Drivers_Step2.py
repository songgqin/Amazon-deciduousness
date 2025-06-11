# Baic Color Chart for Bivariate Choropleth Map,
#       p3(Color3) ^------------ p4(Color4)
#                  .           .
#                  .           .
#        var1      .           .
#                  .           .
#                  .           .
#       p1(Color1) ------------> p2(Color2)
#                      var2
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

# Baic Color Chart for Ternary Choropleth Map, in the order of p1, p2, p3
#                      p1(Var1, Color1)
#                      /\
#                     /  \
#    p2(Var2, Color2)/____\p3(Var3, Color3)
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

# In[]
import numpy as np
import itertools


# Function that maps two variables to an RGB color array of corresponding color chart.
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


# Function that get small rectangles and corresponding colorWeights
def getLegend2(atla, n):
    # Generate the grid
    AA, BB = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    XX, YY = np.meshgrid(np.linspace(-1, 1, n + 1), np.linspace(-1, 1, n + 1))
    # Call getCMap2 function (define it separately)
    tCMap = getCMap2(AA, BB, atla, n)

    rectangles = []
    colorWeights = []
    for j in range(n):
        for i in range(n):
            rectangles.append([(XX[i, j], YY[i, j]), (XX[i + 1, j], YY[i + 1, j]), (XX[i + 1, j + 1], YY[i + 1, j + 1]),
                               (XX[i, j + 1], YY[i, j + 1])])
            colorWeights.append([tCMap[i, j, 0], tCMap[i, j, 1], tCMap[i, j, 2]])

    return rectangles, colorWeights


# Function that maps three variables to an RGB color array of corresponding color chart.
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
    # ABC = ABC / np.sum(ABC, axis=1).reshape((-1,1))
    # ABC[np.isnan(ABC)] = 1/3
    # ABC = ABC / np.max(ABC, axis=1).reshape((-1,1))
    # ABC[np.isnan(ABC)] = 1

    colorABC = (atla[0, :].reshape((1, -1))) * (ABC[:, 0].reshape((-1, 1))) + \
               (atla[1, :].reshape((1, -1))) * (ABC[:, 1].reshape((-1, 1))) + \
               (atla[2, :].reshape((1, -1))) * (ABC[:, 2].reshape((-1, 1)))
    # colorABC = colorABC / np.max(colorABC, axis=1).reshape((-1,1))
    colorABC[np.isnan(colorABC)] = 1
    CMap = np.zeros((A.shape[0], A.shape[1], 3))
    CMap[:, :, 0] = np.reshape(colorABC[:, 0], A.shape)
    CMap[:, :, 1] = np.reshape(colorABC[:, 1], A.shape)
    CMap[:, :, 2] = np.reshape(colorABC[:, 2], A.shape)

    return CMap


# Function that get small triangles and corresponding colorWeights
def getLegend3(atla, n):
    # the triangle legend would be like below
    # three points are (1/2,√3/2),(0,0),(1,0)
    # the small triangle according to the order
    # of uper to lower and left to right
    #                      p1(Var1, Color1)
    #                      /\
    #                     /  \
    #    p2(Var2, Color2)/____\p3(Var3, Color3)

    p1, p2, p3 = (1 / 2, np.sqrt(3) / 2), (0, 0), (1, 0)
    YList = np.linspace(p1[1], p2[1], n + 1)
    XList_L = np.linspace(p1[0], p2[0], n + 1)
    XList_R = np.linspace(p1[0], p3[0], n + 1)

    triangles = []
    colorWeights = []
    for i in range(n):  # Y rows
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


# In[]
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
from S1_Data_Preprocessing_amazon import readTif_gdal  # , save_tif
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

matplotlib.use("Qt5Agg")
matplotlib.rcParams['figure.dpi'] = 200
import numpy.ma as ma

# 导入Cartopy专门提供的经纬度的Formatter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# 导入底图包
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams["axes.unicode_minus"] = True  # 显示负号
matplotlib.use('Qt5Agg')
matplotlib.rcParams['figure.dpi'] = 300
cm2in = 1 / 2.54  # centimeters in inches


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


# In[] Visualization of the dominant drivers
# dominant_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\Feature_importance\Dominant_asynchrony_shap_feature.tif'
# dominant_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig4_driver\Dominant_asynchrony_9feautures.tif'
dominant_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig4_driver\Dominant_four_class_herbivory_0521.tif'
_geo, _proj, dominant = readTif_gdal(dominant_path)

precipitation_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Drivers_Climate_Variables\Visualization_Variables\hydroclimate_precipitation_ERA_visualized.tif'
_,_,rainfall = readTif_gdal(precipitation_path)

np.nanmean(rainfall[dominant==3])

# In[]

color_list = [
    'tab:orange',  # PAR
    'tab:blue',  # Hydroclimate
    'lightgray',  # Soil
    'tab:brown',  # Herbivory
]
custom_cmap = mcolors.ListedColormap(color_list)

# feature_name = ['PAR', 'MCWD', 'DSL', 'MAP', 'VPD', 'Soil Clay',
#                 'Soil Fertility', 'Soil Sand', 'Canopy Height', 'Isohydricity']

# feature_name = ['PAR', 'MAP', 'VPD', 'Soil Clay', 'Soil Fertility', 'Soil Moisture', 'Soil Sand',
#                 'Canopy Height', 'Isohydricity']  # 'Temperature',

# feature_name = ['PAR', 'Hydroclimate', 'Soil', 'Vegetation', 'Herbivory', ]

feature_name = ['Light', 'Hydroclimate', 'Soil', 'Herbivory', ]

# Proportion of the area
dominant_res = dominant[~np.isnan(dominant)]
unique, counts = np.unique(dominant_res, return_counts=True)
counts_pro = counts / np.sum(counts)

re_rank = np.argsort(-counts_pro)

re_rank_name = [feature_name[i] for i in re_rank]
re_rank_color = [color_list[i] for i in re_rank]
re_rank_counts = [counts_pro[i] for i in re_rank]

# In[]
fig = plt.figure(dpi=300,figsize=(7,6))

##### main axis
left, bottom, width, height = 0, 0.08, 0.98, 0.84  # the position of left lower corner, width and height of the figure

proj = ccrs.PlateCarree()
ax = fig.add_subplot([left, bottom, width, height], projection=proj)

# set land and ocean features
ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
ax.add_feature(cfeature.OCEAN, zorder=2)
# add amazon basin boundary
add_shp(ax, lw=0.5, ec='k', fc='none', zorder=3)

plt.imshow(dominant, origin='upper', cmap=custom_cmap,
           extent=[-79.77497863776252, -44.516103736085974, -20.521922834169924, 8.628408135496414],
           transform=ccrs.PlateCarree(), zorder=2)


# draw a white rectangle around the subplots
left, bottom, width, height = 0.690, 0.10, 0.25, 0.25
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

left, bottom, width, height = 0.70, 0.15, 0.22, 0.20
ax1 = fig.add_axes([left, bottom, width, height])
# bar plot to show the proportion of the area
ax1.bar(re_rank_name, re_rank_counts, color=re_rank_color, edgecolor='black')
ax1.set_ylim(0, 0.85)
ax1.yaxis.set_ticks_position('right')
ax1.yaxis.set_label_position('right')
ax1.set_ylabel('Proportion of the area', fontsize=8)
ax1.set_xlabel('Dominant drivers', fontsize=8)
ax1.set_xticklabels(re_rank_name, ha='center')
ax1.tick_params(axis='both', which='both', colors='k', left=False, right=True, top=False, bottom=True,
                labelleft=False, labelright=True, labeltop=False, labelbottom=True, labelsize=6)

for p in ax1.patches:
    # 计算标签的x/y坐标（柱子中心偏上）
    x = p.get_x() + p.get_width() / 2  # 水平居中
    y = p.get_height() + 0.02  # 垂直位置：高度+偏移量
    # 3. 添加文本（格式化为整数，居中对齐）
    ax1.text(x, y, f'{p.get_height() * 100:.1f}' + "%", ha='center',  # 水平对齐
            va='bottom',  # 垂直对齐
            fontsize=6,
            color='black')

patches = [mpatches.Patch(color=color_list[i], label="{l}".format(l=feature_name[i])) for i in range(len(feature_name))]
# put those patched as legend-handles into the legend
# plt.legend(handles=patches, bbox_to_anchor=(1.12, 0.45), borderaxespad=0.05, facecolor='white', framealpha=1,fontsize=10)
# plt.legend(handles=patches, bbox_to_anchor=(0.99, 0.20), borderaxespad=0.05, facecolor='white', framealpha=1, fontsize=8)
ax.legend(handles=patches, bbox_to_anchor=(0.5, -0.10), loc='lower center', borderaxespad=0.05, facecolor='white', frameon=False, fontsize=10, ncol=5)


# fig.legend(
#     handles=lns,
#     labels=[line.get_label() for line in lns],
#     loc='lower center',
#     bbox_to_anchor=(0.5, -0.05),
#     ncol=3,
#     columnspacing=1.5,
#     frameon=False,
#     borderpad=1.2 , # 图例边框内边距,
#     fontsize=20
# )

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

ax.tick_params(axis='both', labelsize=10, direction='out', colors='k', length=3, width=0.9, which='major',
               left=True, right=False, top=True, bottom=False,
               labelleft=True, labelright=False, labeltop=True, labelbottom=False, zorder=10)

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig4_driver\Dominant_asynchrony_herbivory_four_class_0521.png'
fig.savefig(save_path, dpi=300, bbox_inches='tight')

# In[] proportion of the dominant drivers across each subregion
from shapely.wkt import loads as load_wkt
import rasterio
import rasterio.mask
from osgeo import ogr

shp_file = ogr.Open(r"J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\ThreeRegions_Boundary\clip\Amazon_ThreeRegions_Clip.shp")
layer = shp_file.GetLayer()

polygons = [feature.GetGeometryRef().ExportToWkt() for feature in layer]
polygons_name = [feature.GetField('Name') for feature in layer]

# Convert WKT polygons to Shapely geometries
shp_polygon = [load_wkt(polygon) for polygon in polygons]

# feature_name = ['PAR', 'MAP', 'VPD', 'Soil Clay', 'Soil Fertility', 'Soil Moisture', 'Soil Sand',
#                 'Canopy Height', 'Isohydricity']  # 'Temperature',
feature_name = ['PAR', 'Hydroclimate', 'Soil', 'Vegetation', 'VPD', ]

# In[]

# dominant_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig4_driver\Dominant_asynchrony_9feautures.tif'
dominant_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig4_driver\Dominant_five_class_herbivory.tif'

reduce_rmse = [8.4]
corresponding_gpp = [2.1]

with rasterio.open(dominant_path) as src:
    for i, polygon in enumerate(shp_polygon):
        # Mask the TIFF with the polygon
        out_image, out_transform = rasterio.mask.mask(src, [polygon], crop=True)
        out_image = out_image[0]
        out_image = out_image.astype(np.float32)
        out_image_res = out_image[~np.isnan(out_image)]
        # Calculate the proportion of each feature
        unique, counts = np.unique(out_image_res, return_counts=True)
        counts_pro = counts / np.sum(counts)
        # print(polygons_name[i], 'PAR: {:.3f}'.format(np.sum(out_image_res == 0) / len(out_image_res)),
        #       'Hydroclimate: {:.3f}'.format(np.sum(out_image_res == 1) / len(out_image_res)),
        #       'Soil: {:.3f}'.format(np.sum(out_image_res == 2) / len(out_image_res)),
        #       'Vegetation: {:.3f}'.format(np.sum(out_image_res == 3) / len(out_image_res)),
        #       'VPD: {:.3f}'.format(np.sum(out_image_res == 4) / len(out_image_res))
        #       )
        print(polygons_name[i],'Hydroclimate: {:.3f}, PAR: {:.3f}, Soil: {:.3f}, VPD: {:.3f}'.format(counts_pro[1], counts_pro[0], counts_pro[2], counts_pro[4]))

# In[]
dominant_res = dominant[~np.isnan(dominant)]
unique, counts = np.unique(dominant_res, return_counts=True)
counts_pro = counts / np.sum(counts)

print('Proportion of the area Hydroclimate: {:.3f}, PAR: {:.3f}, Soil: {:.3f}, Herbivory: {:.3f}'.format(counts_pro[1], counts_pro[0], counts_pro[2], counts_pro[3]))

# In[] Transfer the npy data to tif data
# npy_path = r'C:\Users\96940\OneDrive\桌面\herbivory\her_woody_Full_Amazon.npy'
#
# herbivory_map = np.load(npy_path)
#
# ref_map_path = r'C:\Users\96940\OneDrive\桌面\herbivory\MODIS_Global_LandCover_2020_5m.tif'
# _geo_5m, _proj_5m, cls_map = readTif_gdal(ref_map_path)
#
# save_path = r'C:\Users\96940\OneDrive\桌面\herbivory\herbivory_map_5m_full_Woody.tif'
# save_tif(herbivory_map, save_path, _geo_5m, _proj_5m, 1)
