from skimage import morphology
# from S1_Data_Preprocessing import readTif_gdal, save_tif
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import xgboost as xgb
import cv2
import matplotlib
from osgeo import gdal
import seaborn as sns
import pandas as pd
import shap
import copy
# from pygam import LinearGAM, s, f, te
from tqdm import tqdm

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


def normalize(data):
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_nor = np.divide((data - data_min), (data_max - data_min), out=np.zeros_like(data),
                         where=(data_max - data_min) != 0)
    # data_nor = (data - data_min) / (data_max - data_min)
    return data_nor


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

# r2 = np.corrcoef(y_test.reshape(-1), y_pred)[0, 1]**2
# In[] Read the climate variables

SIF_data_path_1 = r'..\Main_Figures\Fig3\Data\Amazon_GOSIF_mean.tif'
SIF_data_path_2 = r'..\Main_Figures\Fig3\Data\Global_CSIF\CSIF_Amazon.TIF'
EVI_data_path = r'..\Main_Figures\Fig3\Data\Input_Variables\BRDF_EVI.tif'

VPDecM_path =  r'..Main_Figures\Fig3\Data\VPM_Model\GPP\LD_LUE_GPP_Amazon.tif'


VPM_path0 = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\GLASS_GPP\GLASS_GPP_Amazon.TIF' # EC-LUE
VPM_path1 = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\Two_leaves_LUE\Two_Leaves_GPP_Amazon_2018.TIF'
VPM_path2 = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\GPP\MOD_LUE_GPP.tif'

VPM_GPP_geo, VPM_GPP_prj, VPM_GPP0 = readTif_gdal(VPM_path0) # EC-LUE # > 1000 is nan
VPM_GPP_geo, VPM_GPP_prj, VPM_GPP1 = readTif_gdal(VPM_path1) # Two_leaves_LUE # 0 is nan
VPM_GPP_geo, VPM_GPP_prj, VPM_GPP2 = readTif_gdal(VPM_path2) # MOD-LUE

VPM_GPP0 = np.array(VPM_GPP0, dtype=np.float32)
VPM_GPP1 = np.array(VPM_GPP1, dtype=np.float32)
VPM_GPP2 = np.array(VPM_GPP2, dtype=np.float32)

VPM_GPP0[VPM_GPP0 == 65535] = np.nan
VPM_GPP1[VPM_GPP1 == 0] = np.nan

VPM_GPP0 = cv2.resize(VPM_GPP0, (raws_y, columns_x))
VPM_GPP1 = cv2.resize(VPM_GPP1, (raws_y, columns_x))
VPM_GPP2 = cv2.resize(VPM_GPP2, (raws_y, columns_x))

# VPM_GPP = copy.deepcopy(VPM_GPP1)

VPM_GPP = np.nanmean(np.stack([VPM_GPP0, VPM_GPP1, VPM_GPP2], axis=2), axis=2) # essemble the 3LUE-EC GPP Model

# VPM_GPP = np.nanmean(np.stack([VPM_GPP0, VPM_GPP2], axis=2), axis=2) # essemble the VPM GPP

LUE_data = cv2.resize(VPM_GPP, (raws_y, columns_x)) # Three ensemble LUE average

# _geoLUE, _prjLUE, LUE_data = readTif_gdal(LUE_data_path)
_geoSIF, _prjSIF, SIF_data_1 = readTif_gdal(SIF_data_path_1)
_, _, SIF_data_2 = readTif_gdal(SIF_data_path_2)
_geoEVI, _prjEVI, EVI_data = readTif_gdal(EVI_data_path)
_geoVPDecM, _prjVPDecM, VPDecM_data = readTif_gdal(VPDecM_path)


EVI_data = cv2.resize(EVI_data, (raws_y, columns_x))
VPDecM_data = cv2.resize(VPDecM_data, (raws_y, columns_x))

SIF_data_1 = cv2.resize(SIF_data_1, (raws_y, columns_x))
SIF_data_2 = cv2.resize(SIF_data_2, (raws_y, columns_x))


SIF_data_1_nor = (SIF_data_1-np.nanmin(SIF_data_1,axis=2)[:, :, np.newaxis])/(np.nanmax(SIF_data_1,axis=2)-np.nanmin(SIF_data_1,axis=2))[:, :, np.newaxis]
SIF_data_2_nor = (SIF_data_2-np.nanmin(SIF_data_2,axis=2)[:, :, np.newaxis])/(np.nanmax(SIF_data_2,axis=2)-np.nanmin(SIF_data_2,axis=2))[:, :, np.newaxis]

SIF_data = (SIF_data_2_nor+SIF_data_2_nor)/2

# In[] Explore the relationship between LUE and SIF, LUE*(1-Dec) and SIF

forest_mask_res = forest_mask.reshape(-1)

LUE_data_res = LUE_data.reshape(-1, 12)
SIF_data_res = SIF_data.reshape(-1, 12)
EVI_data_res = EVI_data.reshape(-1, 12)
VPDecM_data_res = VPDecM_data.reshape(-1, 12)

VPM_GPP0_res = VPM_GPP0.reshape(-1, 12) #EC-LUE
VPM_GPP1_res = VPM_GPP1.reshape(-1, 12) #VPM
VPM_GPP2_res = VPM_GPP2.reshape(-1, 12) #MOD-LUE


LUE_data_res_forest = LUE_data_res[forest_mask_res, :]
SIF_data_res_forest = SIF_data_res[forest_mask_res, :]
EVI_data_res_forest = EVI_data_res[forest_mask_res, :]
VPDecM_data_res_forest = VPDecM_data_res[forest_mask_res, :]

VPM_GPP0_res_forest = VPM_GPP0_res[forest_mask_res, :] #EC-LUE
VPM_GPP1_res_forest = VPM_GPP1_res[forest_mask_res, :] #VPM
VPM_GPP2_res_forest = VPM_GPP2_res[forest_mask_res, :] #MOD-LUE

# In[]
# np to save the r2 for each pixel
r2_LUE_SIF = np.zeros((LUE_data_res_forest.shape[0])) * np.nan  # Ensemble LUE and SIF
r2_LA_SIF = np.zeros((LUE_data_res_forest.shape[0])) * np.nan  # LA-LUE and SIF

r2_VPM_0_SIF = np.zeros((LUE_data_res_forest.shape[0])) * np.nan  # EC-LUE GPP and SIF
r2_VPM_1_SIF = np.zeros((LUE_data_res_forest.shape[0])) * np.nan  # VPM GPP and SIF
r2_VPM_2_SIF = np.zeros((LUE_data_res_forest.shape[0])) * np.nan  # MOD-LUE GPP and SIF

# np to save the RMSE for each pixel
rmse_LUE_SIF = np.zeros((LUE_data_res_forest.shape[0])) * np.nan  # Ensemble LUE and SIF
rmse_LA_SIF = np.zeros((LUE_data_res_forest.shape[0])) * np.nan  #LA-LUE and SIF
rmse_VPM_0_SIF = np.zeros((LUE_data_res_forest.shape[0])) * np.nan  # EC-LUE GPP and SIF
rmse_VPM_1_SIF = np.zeros((LUE_data_res_forest.shape[0])) * np.nan  # VPM GPP and SIF
rmse_VPM_2_SIF = np.zeros((LUE_data_res_forest.shape[0])) * np.nan  # MOD-LUE GPP and SIF

x_list = np.arange(12)

new_GPP_dec_res = np.zeros((LUE_data_res_forest.shape[0], 12)) * np.nan

for i in tqdm(range(LUE_data_res_forest.shape[0])):
    x_raw_list = ~np.isnan(LUE_data_res_forest[i]) & ~np.isnan(SIF_data_res_forest[i]) & ~np.isnan(VPDecM_data_res_forest[i])
    x_raw_month = x_list[x_raw_list]
    # r2_LUE_SIF[i] = np.corrcoef(LUE_data_res_forest[i, x_raw_month], SIF_data_res_forest[i, x_raw_month])[0, 1]  # ** 2
    # r2_LA_SIF[i] = np.corrcoef(LUE_data_res_forest[i, x_raw_month] * (1 - Dec_data_res_forest_nor[i, x_raw_month]),SIF_data_res_forest[i, x_raw_month])[0, 1]
    # rmse_LUE_SIF[i] = mean_squared_error(LUE_data_res_forest[i, x_raw_month], SIF_data_res_forest[i, x_raw_month])
    try:
        nor_LUE_month = normalize(LUE_data_res_forest[i, x_raw_month])
        nor_SIF_month = normalize(SIF_data_res_forest[i, x_raw_month])
        nor_LA_LUE_month = normalize(VPDecM_data_res_forest[i, x_raw_month])
        nor_VPM_GPP0_month = normalize(VPM_GPP0_res_forest[i, x_raw_month])
        nor_VPM_GPP1_month = normalize(VPM_GPP1_res_forest[i, x_raw_month])
        nor_VPM_GPP2_month = normalize(VPM_GPP2_res_forest[i, x_raw_month])

        # nor_LA_LUE_month = np.roll(nor_LA_LUE_month, -1)  # leaf age is corresponding to the next month GPP
        rmse_LUE_SIF[i] = np.sqrt(mean_squared_error(nor_SIF_month, nor_LUE_month))
        rmse_LA_SIF[i] = np.sqrt(mean_squared_error(nor_SIF_month, nor_LA_LUE_month))
        rmse_VPM_0_SIF[i] = np.sqrt(mean_squared_error(nor_SIF_month, nor_VPM_GPP0_month))
        rmse_VPM_1_SIF[i] = np.sqrt(mean_squared_error(nor_SIF_month, nor_VPM_GPP1_month))
        rmse_VPM_2_SIF[i] = np.sqrt(mean_squared_error(nor_SIF_month, nor_VPM_GPP2_month))

        r2_LUE_SIF[i] = np.corrcoef(nor_LUE_month, nor_SIF_month)[0, 1]
        r2_LA_SIF[i] = np.corrcoef(nor_LA_LUE_month, nor_SIF_month)[0, 1]
        r2_VPM_0_SIF[i] = np.corrcoef(nor_VPM_GPP0_month, nor_SIF_month)[0, 1]
        r2_VPM_1_SIF[i] = np.corrcoef(nor_VPM_GPP1_month, nor_SIF_month)[0, 1]
        r2_VPM_2_SIF[i] = np.corrcoef(nor_VPM_GPP2_month, nor_SIF_month)[0, 1]

    except:
        pass

# In[]
delat_r2 = copy.deepcopy(r2_LA_SIF) - copy.deepcopy(r2_LUE_SIF)
# delta_rmse = (rmse_LUE_SIF - rmse_LA_SIF)/ rmse_LUE_SIF
delta_rmse = (copy.deepcopy(rmse_LA_SIF) - copy.deepcopy(rmse_LUE_SIF)) / copy.deepcopy(rmse_LUE_SIF)


# print(np.nanmean(delat_r2))
print(np.nansum(delat_r2[~np.isnan(delat_r2)] >= 0) / np.nansum(~np.isnan(delat_r2)))
print(np.nansum(delta_rmse[~np.isnan(delta_rmse)] <= 0) / np.nansum(~np.isnan(delta_rmse)))
proportion = np.nansum(delat_r2[~np.isnan(delat_r2)] >= 0) / np.nansum(~np.isnan(delat_r2))

delta_r2_vpm0 = copy.deepcopy(r2_LA_SIF) - copy.deepcopy(r2_VPM_0_SIF)
delta_r2_vpm1 = copy.deepcopy(r2_LA_SIF) - copy.deepcopy(r2_VPM_1_SIF)
delta_r2_vpm2 = copy.deepcopy(r2_LA_SIF) - copy.deepcopy(r2_VPM_2_SIF)

delta_rmse_vpm0 = (copy.deepcopy(rmse_LA_SIF) - copy.deepcopy(rmse_VPM_0_SIF)) / copy.deepcopy(rmse_VPM_0_SIF)
delta_rmse_vpm1 = (copy.deepcopy(rmse_LA_SIF) - copy.deepcopy(rmse_VPM_1_SIF)) / copy.deepcopy(rmse_VPM_1_SIF)
delta_rmse_vpm2 = (copy.deepcopy(rmse_LA_SIF) - copy.deepcopy(rmse_VPM_2_SIF)) / copy.deepcopy(rmse_VPM_2_SIF)

print(np.nansum(delta_rmse_vpm0[~np.isnan(delta_rmse_vpm0)] <= 0) / np.nansum(~np.isnan(delta_rmse_vpm0)))
print(np.nansum(delta_rmse_vpm1[~np.isnan(delta_rmse_vpm1)] <= 0) / np.nansum(~np.isnan(delta_rmse_vpm1)))
print(np.nansum(delta_rmse_vpm2[~np.isnan(delta_rmse_vpm2)] <= 0) / np.nansum(~np.isnan(delta_rmse_vpm2)))

print(np.nansum(delta_r2_vpm0[~np.isnan(delta_r2_vpm0)] >= 0) / np.nansum(~np.isnan(delta_r2_vpm0)))
print(np.nansum(delta_r2_vpm1[~np.isnan(delta_r2_vpm1)] >= 0) / np.nansum(~np.isnan(delta_r2_vpm1)))
print(np.nansum(delta_r2_vpm2[~np.isnan(delta_r2_vpm2)] >= 0) / np.nansum(~np.isnan(delta_r2_vpm2)))

# In[] Visualization of delta r2 and delta rmse

r2_map = np.zeros((columns_x, raws_y)) * np.nan
r2_map[forest_mask] = delat_r2


delta_rmse_map = np.zeros((columns_x, raws_y)) * np.nan
delta_rmse_map[forest_mask] = delta_rmse

save_path = r'..\Main_Figures\Fig3\Delta_r2_LA_Ensemble_Two_SIF_basin.tif'
save_tif(r2_map, save_path,  _geoVPDecM, _prjVPDecM,1)

save_path = r'..\Main_Figures\Fig3\Delta_rmse_LA_Ensemble_Two_SIF_basin.tif'
save_tif(delta_rmse_map, save_path, _geoVPDecM, _prjVPDecM, 1)

r2_LUE_SIF_map = np.zeros((columns_x, raws_y)) * np.nan
r2_LA_SIF_map = np.zeros((columns_x, raws_y)) * np.nan

r2_LUE_SIF_map[forest_mask] = r2_LUE_SIF
r2_LA_SIF_map[forest_mask] = r2_LA_SIF

rmse_LUE_SIF_map = np.zeros((columns_x, raws_y)) * np.nan
rmse_LA_SIF_map = np.zeros((columns_x, raws_y)) * np.nan

rmse_LUE_SIF_map[forest_mask] = rmse_LUE_SIF
rmse_LA_SIF_map[forest_mask] = rmse_LA_SIF


# r2 map and rmse map

r2_VPM_0_SIF_map = np.zeros((columns_x, raws_y)) * np.nan
r2_VPM_1_SIF_map = np.zeros((columns_x, raws_y)) * np.nan
r2_VPM_2_SIF_map = np.zeros((columns_x, raws_y)) * np.nan

r2_VPM_0_SIF_map[forest_mask] = r2_VPM_0_SIF
r2_VPM_1_SIF_map[forest_mask] = r2_VPM_1_SIF
r2_VPM_2_SIF_map[forest_mask] = r2_VPM_2_SIF

# rmse map
rmse_VPM_0_SIF_map = np.zeros((columns_x, raws_y)) * np.nan
rmse_VPM_1_SIF_map = np.zeros((columns_x, raws_y)) * np.nan
rmse_VPM_2_SIF_map = np.zeros((columns_x, raws_y)) * np.nan
rmse_VPM_0_SIF_map[forest_mask] = rmse_VPM_0_SIF
rmse_VPM_1_SIF_map[forest_mask] = rmse_VPM_1_SIF
rmse_VPM_2_SIF_map[forest_mask] = rmse_VPM_2_SIF

save_path_0 = r'..\Main_Figures\Fig3\LUE_Individual\EC_LUE_r2_SIF.tif'
save_path_1 = r'..\Main_Figures\Fig3\LUE_Individual\TL_LUE_r2_SIF.tif'
save_path_2 = r'..\Main_Figures\Fig3\LUE_Individual\MOD_LUE_r2_SIF.tif'

save_tif(r2_VPM_0_SIF_map, save_path_0, _geoVPDecM, _prjVPDecM, 1)
save_tif(r2_VPM_1_SIF_map, save_path_1, _geoVPDecM, _prjVPDecM, 1)
save_tif(r2_VPM_2_SIF_map, save_path_2, _geoVPDecM, _prjVPDecM, 1)

save_path_0 = r'..\Main_Figures\Fig3\LUE_Individual\EC_LUE_rmse_SIF.tif'
save_path_1 = r'..\Main_Figures\Fig3\LUE_Individual\TL_LUE_rmse_SIF.tif'
save_path_2 = r'..\Main_Figures\Fig3\LUE_Individual\MOD_LUE_rmse_SIF.tif'

save_tif(rmse_VPM_0_SIF_map, save_path_0, _geoVPDecM, _prjVPDecM, 1)
save_tif(rmse_VPM_1_SIF_map, save_path_1, _geoVPDecM, _prjVPDecM, 1)
save_tif(rmse_VPM_2_SIF_map, save_path_2, _geoVPDecM, _prjVPDecM, 1)

save_path_3 = r'..\Main_Figures\Fig3\LUE_Individual\Ensemble_LUE_rmse_SIF.tif'
save_tif(rmse_LUE_SIF_map, save_path_3, _geoVPDecM, _prjVPDecM, 1)
save_path_4 = r'..\Main_Figures\Fig3\LUE_Individual\LA_LUE_rmse_SIF.tif'
save_tif(rmse_LA_SIF_map, save_path_4, _geoVPDecM, _prjVPDecM, 1)

save_path_5 = r'..\Main_Figures\Fig3\LUE_Individual\Ensemble_LUE_r2_SIF.tif'
save_tif(r2_LUE_SIF_map, save_path_5, _geoVPDecM, _prjVPDecM, 1)

save_path_6 = r'..\Main_Figures\Fig3\LUE_Individual\LA_LUE_r2_SIF.tif'
save_tif(r2_LA_SIF_map, save_path_6, _geoVPDecM, _prjVPDecM, 1)
