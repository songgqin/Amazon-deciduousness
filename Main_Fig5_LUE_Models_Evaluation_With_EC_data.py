import os
import copy
import cv2
from osgeo import gdal
# from patchify import patchify
import matplotlib
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde
# from S1_Data_Preprocessing_amazon import save_tif
from skimage import morphology
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from scipy import signal
from scipy.signal import savgol_filter
from scipy import interpolate
from S7_SG_Smooth import sgfilter_line
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import odeint
import copy
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
import matplotlib
import time

plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams["axes.unicode_minus"] = True  # 显示负号
matplotlib.use('Qt5Agg')
matplotlib.rcParams['figure.dpi'] = 300



# In[] functions

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


# extract leaf age dynamic


def leaf_demography_model(dec_data_site, LAI_total_max=6, Ty_to_M=2, TM_to_O=4):
    # litterfall data
    LAI_lifferfall = LAI_total_max * dec_data_site
    # LAI canopy seasonality
    LAI = LAI_total_max * (1 - dec_data_site)
    # Flush data = delta LAI + LAI_lifferfall; delta LAI = LAI_t+1 - LAI_t; using the repeat LAI to extract the flushing LAI
    delta_LAI = copy.deepcopy(LAI) * np.nan  # delta LAI
    delta_LAI[:-1] = np.diff(LAI)
    delta_LAI[-1] = LAI[0] - LAI[-1]
    flush_data = np.maximum(delta_LAI + LAI_lifferfall, 0)

    def leaf_age_dynamic_model(LAIy0, LAIm0):
        # 定义输入数据
        LAI_total = np.append(LAI, LAI)
        Flush = np.append(flush_data, flush_data)

        months = len(LAI_total)

        # 初始化LAI分类
        LAIY = np.zeros(months)
        LAIM = np.zeros(months)
        LAIO = np.zeros(months)
        simulated_litterfall = np.zeros(months)

        # 初始条件以及限制条件
        LAIY[0] = LAIy0
        LAIM[0] = LAIm0
        LAIO[0] = np.maximum(LAI_total[0] - LAIy0 - LAIm0, 1e-3)

        # 通过微分方程计算增量
        for t in range(months - 1):
            dLAIy = Flush[t] - LAIY[t] / Ty_to_M
            dLAIm = LAIY[t] / Ty_to_M - LAIM[t] / TM_to_O

            # 先计算未调整的下一状态
            LAIY_next = LAIY[t] + dLAIy
            LAIM_next = LAIM[t] + dLAIm
            LAIO_next = LAI_total[t + 1] - LAIY_next - LAIM_next  # LAI_total[t + 1]

            # 处理LAIO为负的情况：将超出部分按比例从Y和M中扣除
            if LAIO_next < 1e-3:
                decay_factor = 0.5  # 老叶保留比例，可调参数
                preserved_LAIO = LAIO[t] * decay_factor / TM_to_O
                excess = LAIY_next + LAIM_next + preserved_LAIO - LAI_total[t + 1]

                if excess > 0:
                    # 按比例削减Y和M
                    total_YM = LAIY_next + LAIM_next
                    scale = (LAI_total[t + 1] - preserved_LAIO) / total_YM if total_YM > 0 else 0  #
                    LAIY_next *= scale
                    LAIM_next *= scale
                    LAIO_next = preserved_LAIO
                else:
                    LAIO_next = preserved_LAIO
            # 更新状态
            LAIY[t + 1] = max(LAIY_next, 0)
            LAIM[t + 1] = max(LAIM_next, 0)
            LAIO[t + 1] = max(LAIO_next, 0)
            # 计算落叶量
            simulated_litterfall[t] = LAIO[t] + LAIM[t] / TM_to_O - LAIO[t + 1]

        # 截取后半段数据
        LAIY_final = LAIY[months // 2:]
        LAIM_final = LAIM[months // 2:]
        LAIO_final = LAIO[months // 2:]
        simulated_litterfall_final = simulated_litterfall[months // 2:]

        return LAIY_final, LAIM_final, LAIO_final, simulated_litterfall_final
        # return LAIY, LAIM, LAIO, simulated_litterfall

    # 定义设计目标函数包含物理约束
    def error_minimize(LAIy0, LAIm0):
        # 初始条件检查
        if (LAIy0 + LAIm0 > 6) or (LAIy0 < 0) or (LAIm0 < 0):
            return -1e6  # 初始条件违规，直接返回大惩罚
        # 运行模型
        try:
            _, _, _, simulated = leaf_age_dynamic_model(LAIy0, LAIm0)

            actual = LAI_lifferfall
            # 计算指标
            rmse = np.sqrt(np.mean((simulated - actual) ** 2))
            corr = np.corrcoef(simulated, actual)[0, 1]
            # 动态过程惩罚项：若模拟中出现负值，增加惩罚
            # penalty = np.sum(np.where(simulated < -0.1, 1, 0)) * 10  # 每出现一次负值加10
            return -rmse + 2 * corr  # - penalty

        except Exception as e:
            return -1e6  # 其他错误（如数值不稳定）

    # 根据数据动态设置参数范围
    max_initial_LAI = LAI[0]
    optimizer = BayesianOptimization(
        f=error_minimize,
        pbounds={
            "LAIy0": (0.1, min(5, max_initial_LAI - 0.1)),
            "LAIm0": (0.1, min(5, max_initial_LAI - 0.1))
        },
        allow_duplicate_points=True,
        random_state=42,
        verbose=0
    )
    # 使用Matern核提高对非平滑区域的适应能力
    optimizer.set_gp_params(
        kernel=Matern(nu=2.5),
        alpha=1e-3,
        n_restarts_optimizer=10,
    )

    # 用EI广泛搜索
    optimizer.maximize(
        init_points=5,
        n_iter=50,
        acquisition_function=UtilityFunction(kind="ei", xi=0.1),
        # kappa=5 - 4*(optimizer.space.target.size/55) # 逐渐减小kappa值
    )
    params = optimizer.max['params']

    LAIY, LAIM, LAIO, simulated_litterfall = leaf_age_dynamic_model(params['LAIy0'], params['LAIm0'])

    return LAIY, LAIM, LAIO


# In[]
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

# In[]
# read the eddy flux data cvs file

site_locate_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\Evaluation\eddy_fluxes\Eddy_Fluxes_Sites_Locations.xlsx'
eddy_flux_dir = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\Evaluation\eddy_fluxes\DOY'

VPM_path0 = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\GLASS_GPP\GLASS_GPP_2018_Amazon.TIF'  # EC-LUE
# VPM_path1 = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\ZY_VPM_GPP\VPM_GPP_ZY_Amazon_2019.TIF' # VPM
VPM_path1 = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\Two_leaves_LUE\Two_Leaves_GPP_Amazon_2018.TIF'
# VPM_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\GPP\VPM_LUE_GPP.tif'
VPM_path2 = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\GPP\MOD_LUE_GPP.tif'


VPDecM_path =  r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\GPP\Leaf_Age_EC_GPP3_Wu_emax_v2.tif'

# VPDecM_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\GPP\Leaf_Age_MOD_GPP.tif'
# VPDecM_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\GPP\Leaf_Age_VPM_GPP.tif'

# VPDecM_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\EC_LUE_Dec_GPP.tif'
# VPM_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\VPM_GPP_Xiao.tif'
# VPDecM_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\VPM_Dec_GPP_Xiao.tif'
# VPM_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\BRDF_EVI.tif'
# VPDecM_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Multiyear_mosaic\250m_gapfill\v2_threshold_0.45_overestimate\95qc_DSO_1111\Mean_Filled_95qc\Composite_Data_5km_gf_3y.tif'
Dec_data_path = r'\\GEARS-DC\space3\Song_Amazon_Mapping\Mosaic_Data_Multiyears\Gapfill_based_Mosaic\Multiyear_mosaic\250m_gapfill\v2_threshold_0.45_overestimate\95qc_DSO_1111\Mean_Filled_95qc\Composite_Data_5km_gf_3y.tif'

eddy_flux_name_list = [x for x in os.listdir(eddy_flux_dir) if x.endswith('.csv')]
site_locate_pd = pd.read_excel(site_locate_path)

VPM_GPP_geo, VPM_GPP_prj, VPM_GPP0 = readTif_gdal(VPM_path0)  # EC-LUE # > 1000 is nan
VPM_GPP_geo, VPM_GPP_prj, VPM_GPP1 = readTif_gdal(VPM_path1)  # VPM # 0 is nan
VPM_GPP_geo, VPM_GPP_prj, VPM_GPP2 = readTif_gdal(VPM_path2)  # MOD-LUE

VPM_GPP0 = np.array(VPM_GPP0, dtype=np.float32)
VPM_GPP1 = np.array(VPM_GPP1, dtype=np.float32)
VPM_GPP2 = np.array(VPM_GPP2, dtype=np.float32)

VPM_GPP0[VPM_GPP0 == 65535] = np.nan
VPM_GPP1[VPM_GPP1 == 0] = np.nan

# LA-LUE
VPDecM_GPP_geo, VPDecM_GPP_prj, VPDecM_GPP = readTif_gdal(VPDecM_path)
_geoDec, _prjDec, Dec_data = readTif_gdal(Dec_data_path)

Dec_data = Dec_data.astype(np.float32)
Dec_data[Dec_data > 1000] = np.nan

Dec_data = Dec_data / 1000.0
Dec_data_t_nor = Dec_data - np.nanmin(Dec_data, axis=2)[:, :, np.newaxis]

VPDecM_GPP = np.array(VPDecM_GPP, dtype=np.float32)
# VPDecM_GPP[VPDecM_GPP > 1000] = np.nan
# VPDecM_GPP = np.array(VPDecM_GPP, dtype=np.float32)
# VPDecM_GPP[VPDecM_GPP > 1000] = np.nan
# VPDecM_GPP = VPDecM_GPP / 1000.0
# VPDecM_GPP = (VPDecM_GPP - np.nanmin(VPDecM_GPP, axis=2)[:, :, np.newaxis])
# VPDecM_GPP = 1-VPDecM_GPP
raws_y, columns_x = 786, 650

# VPM_GPP = cv2.resize(VPM_GPP, (raws_y, columns_x))
VPM_GPP0 = cv2.resize(VPM_GPP0, (raws_y, columns_x))
VPM_GPP1 = cv2.resize(VPM_GPP1, (raws_y, columns_x))
VPM_GPP2 = cv2.resize(VPM_GPP2, (raws_y, columns_x))

VPM_GPP = np.nanmean(np.stack([VPM_GPP0, VPM_GPP1, VPM_GPP2], axis=2), axis=2)  # essemble the VPM GPP
# VPM_GPP = np.nanmean(np.stack([VPM_GPP0, VPM_GPP2], axis=2), axis=2) # essemble the VPM GPP

VPDecM_GPP = cv2.resize(VPDecM_GPP, (raws_y, columns_x))
# VPM_GPP = VPM_GPP/10000.0
# VPDecM_GPP = VPDecM_GPP / 1000.0
# VPDecM_GPP = VPDecM_GPP - np.nanmin(VPDecM_GPP, axis=2)[:, :, np.newaxis]
# VPM_GPP[~forest_mask] = np.nan
# VPDecM_GPP[~forest_mask] = np.nan

# In[]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

GEP_all_list = []
VPM_all_list = []
VPDecM_all_list = []

r2_VPM_all = []
r2_VPDecM_all = []
r2_VPM_Dec_all = []

rmse_VPM_all = []
rmse_VPDecM_all = []

eddy_flux_name_list = ['K34_CfluxBF_DOY.csv', 'K67_CfluxBF_DOY.csv', 'CAX_CfluxBF_DOY.csv', 'RJA_CfluxBF_DOY.csv']

month_label = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

plt.subplots_adjust(bottom=0.15, top=0.95, wspace=0.4, hspace=0.5)

for i, eddy_flux_name in enumerate(eddy_flux_name_list):
    eddy_flux_site_name = eddy_flux_name.split('_')[0]
    eddy_flux_site_locate = site_locate_pd[site_locate_pd['Site'] == eddy_flux_site_name]
    eddy_flux_lat = eddy_flux_site_locate['Latitude'].to_numpy()[0]
    eddy_flux_lon = eddy_flux_site_locate['Longitude'].to_numpy()[0]

    pd_eddy_flux_doy = pd.read_csv(os.path.join(eddy_flux_dir, eddy_flux_name))
    pd_eddy_flux_month = pd_eddy_flux_doy.groupby('Month').mean()

    dec_y, dec_x = int((eddy_flux_lon - _geoDec[0]) / _geoDec[1]), int((eddy_flux_lat - _geoDec[3]) / _geoDec[5])

    # using the windows size of 5x5 to extract the dec_month value
    patch_size = 2
    # patch_size = 0
    # if 'K' or 'CAX' not in eddy_flux_name:
    #     patch_size = 9 // 2
    # else:
    #     patch_size = 5 // 2

    VPM_GPP_site_list = np.nanmean(
        VPM_GPP[dec_x - patch_size:dec_x + patch_size + 1, dec_y - patch_size:dec_y + patch_size + 1],
        axis=(0, 1))  # / 365
    VPDecM_GPP_site_list = np.nanmean(
        VPDecM_GPP[dec_x - patch_size:dec_x + patch_size + 1, dec_y - patch_size:dec_y + patch_size + 1],
        axis=(0, 1))  # / 365

    # dec_site_list = np.nanmean(Dec_data_t_nor[dec_x - patch_size:dec_x + patch_size + 1, dec_y - patch_size:dec_y + patch_size + 1], axis=(0, 1))

    # VPDecM_GPP_site_list = VPM_GPP_site_list*(1-dec_site_list)

    month_list = pd_eddy_flux_month.index.to_numpy()
    GEP_list = pd_eddy_flux_month['GEP_smooth'].to_numpy()

    # normalize the VPM and VPDecM GPP, GEP
    VPM_GPP_site_list = (VPM_GPP_site_list - np.nanmin(VPM_GPP_site_list)) / (np.nanmax(VPM_GPP_site_list) - np.nanmin(VPM_GPP_site_list))
    VPDecM_GPP_site_list = (VPDecM_GPP_site_list - np.nanmin(VPDecM_GPP_site_list)) / (np.nanmax(VPDecM_GPP_site_list) - np.nanmin(VPDecM_GPP_site_list))
    GEP_list = (GEP_list - np.nanmin(GEP_list)) / (np.nanmax(GEP_list) - np.nanmin(GEP_list))

    # leaf age GPP should move one month ahead
    # VPDecM_GPP_site_list = np.roll(VPDecM_GPP_site_list, -1) # leaf age GPP should move one month ahead

    # print('Site:', eddy_flux_name.split('_')[0], 'D%:', VPDecM_GPP_site_list)
    # VPDecM_GPP_site_list_nor = (VPDecM_GPP_site_list - np.nanmin(VPDecM_GPP_site_list))
    # VPDecM_GPP_site_list_nor = 1 - VPDecM_GPP_site_list_nor

    # VPDecM_GPP_site_list = copy.deepcopy(VPDecM_GPP_site_list_nor)

    GEP_all_list.append(GEP_list)
    VPM_all_list.append(VPM_GPP_site_list)
    VPDecM_all_list.append(VPDecM_GPP_site_list)

    ax = axes[i // 2, i % 2]

    lns1 = ax.plot(month_list, VPDecM_GPP_site_list, color='#e34a33', linestyle='-',
                   marker='^', markersize=8,
                   markerfacecolor='#e34a33', markeredgecolor='#e34a33',  # 实心上三角
                   label='LD-LUE')

    lns2 = ax.plot(month_list, VPM_GPP_site_list, color='#1f77b4', linestyle='-',
                   marker='s', markersize=8,
                   markerfacecolor='#1f77b4', markeredgecolor='#1f77b4',  # 实心方块
                   label='Ensemble-LUE', alpha=0.8)

    # ax2 = ax.twinx()
    # lns2 = ax2.plot(month_list, VPDecM_GPP_site_list, marker='o', label='VPDecM', c='darkorange')
    # lns4 = ax.plot(month_list, np.array(VPM_GPP_site_list)*np.array(VPDecM_GPP_site_list), marker='^', label='VPM*DEC')
    # ax2.spines['left'].set_position(('outward', 25))
    # ax2.yaxis.set_ticks_position('left')
    # ax2.yaxis.set_label_position('left')

    # remove the nan value in the GPP_list
    nan_mask = np.isnan(GEP_list) | np.isnan(VPM_GPP_site_list) | np.isnan(VPDecM_GPP_site_list)

    r2_VPM = np.corrcoef(VPM_GPP_site_list[~nan_mask], GEP_list[~nan_mask])[0, 1]  # **2
    r2_VPDecM = np.corrcoef(VPDecM_GPP_site_list[~nan_mask], GEP_list[~nan_mask])[0, 1]  # **2

    rmse_VPM = np.sqrt(np.mean((VPM_GPP_site_list[~nan_mask] - GEP_list[~nan_mask]) ** 2))
    rmse_VPDecM = np.sqrt(np.mean((VPDecM_GPP_site_list[~nan_mask] - GEP_list[~nan_mask]) ** 2))
    # r2_VPM_Dec = np.corrcoef(np.array(VPM_GPP_site_list)*np.array(VPDecM_GPP_site_list)[~nan_mask], GEP_list[~nan_mask])[0, 1]  #**2
    # r2_EVI_Dec = np.corrcoef(np.array(VPM_GPP_site_list)*np.array(VPDecM_GPP_site_list)[~nan_mask], VPM_GPP_site_list[~nan_mask])[0, 1]

    r2_VPM_all.append(r2_VPM)
    r2_VPDecM_all.append(r2_VPDecM)

    rmse_VPM_all.append(rmse_VPM)
    rmse_VPDecM_all.append(rmse_VPDecM)
    # r2_VPM_Dec_all.append(r2_VPM_Dec)

    # print(eddy_flux_name.split('_')[0])
    # print('r2_VPM:', r2_VPM)
    # print('r2_VPDecM:', r2_VPDecM)
    # print('r2_EVI_Dec:', r2_EVI_Dec)

    # plt.text(1, 0.3, 'r$^2$ = %.2f' % r2_VPM, fontsize=18)
    # plt.text(1, 0.6, 'r$^2$ = %.2f' % r2_VPDecM, fontsize=18)

    ax.set_ylabel('Normalize GPP', fontsize=16)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0, 0.5, 1], fontsize=14)
    ax.set_xlabel('Month', fontsize=16)
    ax.set_xticks(month_list)
    ax.set_xticklabels(month_label, fontsize=14)

    # ax.tick_params(axis='both', which='major', labelsize=14)

    # ax.legend(loc='upper left')
    # ax1 = ax.twinx()
    # lns3 = ax1.plot(month_list, GEP_list, marker='^', color='k', label='GEP')
    # ax1.set_ylabel('Eddy Flux GEP', fontsize=14)
    # ax1.tick_params(axis='both', which='major', labelsize=14)

    lns3 = ax.plot(month_list, GEP_list, color='k', linestyle='--',
                   marker='o', markersize=8,
                   markerfacecolor='none', markeredgecolor='k',  # 空心圆
                   label='EC observations')
    # ax1.set_ylabel('Eddy Flux GEP', fontsize=14)
    # ax1.tick_params(axis='both', which='major', labelsize=14)

    ax.set_title(eddy_flux_site_name, fontsize=16)
    # ax1.legend(loc='upper right',fontsize=14)
    if i == 0:
        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
    # if i == 3:
    #     ax.legend(lns, labs, fontsize=14, frameon=False)
    del VPM_GPP_site_list, VPDecM_GPP_site_list, month_list, GEP_list

fig.legend(
    handles=lns,
    labels=[line.get_label() for line in lns],
    loc='lower center',
    bbox_to_anchor=(0.5, -0.06),
    ncol=3,
    columnspacing=1.5,
    frameon=False,
    borderpad=1.2,  # 图例边框内边距,
    fontsize=16
)
# plt.tight_layout()
# plt.legend(ax,legend_list, loc='upper left', fontsize=14)
# plt.tight_layout()

# del VPM_GPP_site_list, VPDecM_GPP_site_list, month_list, GEP_list

# show the r2 bar plot of VPM and VPDecM
# ax = axes[2, 2]
# # fig, ax = plt.subplots()
# bar_width = 0.2
# bar_x = np.arange(len(r2_VPM_all))
# ax.bar(bar_x, r2_VPM_all, bar_width, label='VPM')
# ax.bar(bar_x + bar_width, r2_VPDecM_all, bar_width, label='VPDecM')
# # ax.bar(bar_x + 2 * bar_width, r2_VPM_Dec_all, bar_width, label='VPM*DEC')
# ax.set_xticks(bar_x + bar_width)
# ax.set_xticklabels([x.split('_')[0] for x in eddy_flux_name_list], rotation=45)
# ax.set_ylabel('Correlation Coefficient', fontsize=14)  #$^2$
# ax.set_xlabel('Site', fontsize=14)
# ax.legend(fontsize=14, frameon=False, bbox_to_anchor=(1.05, 0.5))

# plt.tight_layout()

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Site_GPP_Comparison_0521.png'
fig.savefig(save_path, dpi=300, bbox_inches='tight')

# In[] bar plot of the r2 of VPM and VPDecM

# lns1 = ax.plot(month_list, VPDecM_GPP_site_list, color='#ff7f0e', linestyle='-',
#                marker='^', markersize=8,
#                markerfacecolor='#ff7f0e', markeredgecolor='black',  # 实心上三角
#                label='LA-LUE')
#
# lns2 = ax.plot(month_list, VPM_GPP_site_list, color='#1f77b4', linestyle='-',
#                marker='s', markersize=8,
#                markerfacecolor='#1f77b4', markeredgecolor='black',  # 实心方块
#                label='Ensemble-LUE')

fig, ax = plt.subplots(figsize=(9, 5))


bar_width = 0.25

n_sites = len(r2_VPM_all)
bar_y = np.arange(n_sites)  # y轴位置基于站点数量生成

# 绘制水平分组条形图
ax.barh(bar_y - bar_width / 2, r2_VPDecM_all[::-1], color='#e34a33',  # VPDecM条形右移半个宽度
        height=bar_width, label='LD-LUE')

ax.barh(bar_y + bar_width / 2, r2_VPM_all[::-1],  # VPM条形左移半个宽度
        height=bar_width, label='Ensemble-LUE', color='#1f77b4', alpha=0.8)

# 设置y轴标签为站点名称
ax.set_yticks(bar_y)  # 主刻度在每组条形中间
ax.set_yticklabels([x.split('_')[0] for x in eddy_flux_name_list[::-1]], ha='right', fontsize=22)  # 旋转45度对齐

ax.set_xticks([-0.5, 0, 0.5, 1])
ax.set_xticklabels([-0.5, 0, 0.5, 1], fontsize=22)
ax.set_xlim(-0.6, 1.1)  # 设置x轴范围

# 标签设置
ax.set_xlabel('Correlation coefficient', fontsize=24, labelpad=15)
ax.set_ylabel('Site', fontsize=24)

# 图例调整到右侧外部
# ax.legend(fontsize=24, frameon=False, ncol=2,
#           columnspacing=1.5,
#           bbox_to_anchor=(0.5, -0.25),  # x=1.25将图例移出绘图区
#           loc='lower center', )

plt.tight_layout()  # 自动优化布局
# plt.show()

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Site_GPP_Comparison_r_Bar_0521.png'
fig.savefig(save_path, dpi=600, bbox_inches='tight')

# In[]

fig, ax = plt.subplots(figsize=(9, 5))
# plt.subplots_adjust(bottom=0.3, top=0.95)

bar_width = 0.25

n_sites = len(rmse_VPDecM_all)
bar_y = np.arange(n_sites)  # y轴位置基于站点数量生成

# 绘制水平分组条形图
ax.barh(bar_y - bar_width / 2, rmse_VPDecM_all[::-1], color='#e34a33',  # VPDecM条形右移半个宽度
        height=bar_width, label='LD-LUE')

ax.barh(bar_y + bar_width / 2, rmse_VPM_all[::-1],  # VPM条形左移半个宽度
        height=bar_width, label='Ensemble-LUE', color='#1f77b4', alpha=0.8)

# 设置y轴标签为站点名称
ax.set_yticks(bar_y)  # 主刻度在每组条形中间
ax.set_yticklabels([x.split('_')[0] for x in eddy_flux_name_list[::-1]], ha='right', fontsize=22)  # 旋转45度对齐

ax.set_xticks([0, 0.25, 0.5])
ax.set_xticklabels([0, 0.25, 0.5], fontsize=22)
ax.set_xlim(-0.1, 0.6)  # 设置x轴范围

# 标签设置
ax.set_xlabel('RMSE', fontsize=24, labelpad=15)
ax.set_ylabel('Site', fontsize=24)

# 图例调整到右侧外部
# ax.legend(fontsize=24, frameon=False, ncol=2,
#           bbox_transform=ax.transAxes,  # 使用坐标轴坐标系
#           columnspacing=1.5,
#           bbox_to_anchor=(0.5, -0.5),  # x=1.25将图例移出绘图区
#           loc='lower center', )

plt.tight_layout()  # 自动优化布局
# plt.show()

save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Site_GPP_Comparison_rmse_Bar_0521.png'
fig.savefig(save_path, dpi=600, bbox_inches='tight')

# In[] scatter plot of the GEP and VPM/VPDecM of all sites

GEP_all_list = np.concatenate(GEP_all_list)
VPM_all_list = np.concatenate(VPM_all_list)
VPDecM_all_list = np.concatenate(VPDecM_all_list)

fig, ax = plt.subplots()
ax.scatter(GEP_all_list, VPM_all_list, label='VPM')

fig, ax = plt.subplots()
ax.scatter(GEP_all_list, VPDecM_all_list, label='VPDecM')

# plt.figure()
# plt.plot(pd_eddy_flux_doy['DOY'], pd_eddy_flux_doy['GEP'])
# plt.plot(pd_eddy_flux_month.index, pd_eddy_flux_month['GEP'])


# In[] read eddy fluxes data from GF site
# eddy_flux_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\Evaluation\eddy_fluxes\ICOSETC_GF-Guy_FLUXNET_DD_L2.csv'
#
# pd_eddy_flux = pd.read_csv(eddy_flux_path)
#
# data_gpp = pd.DataFrame()
# #pd_eddy_flux.iloc['TIMESTAMP','GPP_DT_VUT_MEAN')]
# data_gpp['Time'] = pd_eddy_flux['TIMESTAMP']
# data_gpp['GPP'] = pd_eddy_flux['GPP_DT_VUT_MEAN']
# time_list = data_gpp['Time'].to_list()
#
# time_list = [str(x) for x in time_list]
#
# month_list = [x[4:6] for x in time_list]
#
# data_gpp['Month'] = month_list
#
# data_gpp_month = data_gpp.groupby('Month').mean()
#
# fig, ax = plt.subplots()
# ax.plot(data_gpp_month.index, data_gpp_month['GPP'], marker='o')
# ax.set_xticks(data_gpp_month.index)
# ax.set_xticklabels(data_gpp_month.index.astype(int), rotation=45)
# ax.set_ylabel('GPP')
