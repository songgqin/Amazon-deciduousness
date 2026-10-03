# In[] ==========================================
# 0. 导入环境与全局设置
# ==========================================
import os
import cv2
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from osgeo import gdal
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as mcolors
import warnings
import cartopy.io.shapereader as shpreader
matplotlib.use('Qt5Agg')
warnings.filterwarnings('ignore')



# 全局阈值与配色标准
COLOR_EXP3 = '#e34a33'
COLOR_EXP1 = '#1f77b4'
COLOR_OBS = 'k'


# ==============================================================
# 构建 0.95 极浅灰高级色带
# ==============================================================
# cmap_gray = np.array([[0.95, 0.95, 0.95, 1.0]])  # RGBA for the neutral zone
# cmap_blue = plt.cm.Blues_r(np.linspace(0.15, 0.75, 5))
# cmap_red = plt.cm.Reds(np.linspace(0.15, 0.70, 5))
# colors = np.vstack((cmap_blue, cmap_gray, cmap_red))
# custom_cmap = mcolors.ListedColormap(colors)
# levels = [-1.25, -1.00, -0.75, -0.50, -0.25, -neutral_th, neutral_th, 0.25, 0.50, 0.75, 1.0, 1.25]
# norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=11)
#
# cmap_blue = plt.cm.Blues_r(np.linspace(0.15, 0.75, 5))
# cmap_red = plt.cm.Reds(np.linspace(0.15, 0.70, 5))
# colors = np.vstack((cmap_blue,cmap_red))
# custom_cmap = mcolors.ListedColormap(colors)
# levels = [-1.25, -1.00, -0.75, -0.50, -0.25,0, 0.25, 0.50, 0.75, 1.0, 1.25]
# norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=11)

# In[] ==========================================
# 1. 快速向量化工具函数
# ==========================================
def readTif_gdal_safe(fileName, nbands=12):
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    dataset = gdal.Open(fileName)
    if dataset is None: return None, None, None
    im_width, im_height = dataset.RasterXSize, dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    if im_data.ndim == 3 and im_data.shape[0] <= nbands:
        im_data = np.transpose(im_data, [1, 2, 0])
    im_data = im_data.astype(np.float32)

    im_data[im_data == 65535] = np.nan
    im_data[im_data == -9999] = np.nan
    return dataset.GetGeoTransform(), dataset.GetProjection(), im_data


def normalize_ts(x):
    x_min = np.nanmin(x, axis=1, keepdims=True)
    x_max = np.nanmax(x, axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (x - x_min) / (x_max - x_min)


def vectorized_pearsonr(x, y):
    joint_valid = ~np.isnan(x) & ~np.isnan(y)
    x_safe = np.where(joint_valid, x, np.nan)
    y_safe = np.where(joint_valid, y, np.nan)
    xm = x_safe - np.nanmean(x_safe, axis=1, keepdims=True)
    ym = y_safe - np.nanmean(y_safe, axis=1, keepdims=True)
    r_num = np.nansum(xm * ym, axis=1)
    r_den = np.sqrt(np.nansum(xm ** 2, axis=1) * np.nansum(ym ** 2, axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        return r_num / r_den


def add_shp(ax, shp_path, **kwargs):
    proj = ccrs.PlateCarree()
    reader = shpreader.Reader(shp_path)
    provinces = reader.geometries()
    ax.add_geometries(provinces, proj, **kwargs)
    reader.close()


# In[] ==========================================
# 2. 初始化掩膜与 SIF 验证基准
# ===============================================
print(">>> 正在初始化掩膜与 SIF 验证数据...")
raws_y, columns_x = 786, 650
img_extent = [-79.77, -44.51, -20.52, 8.62]
map_extent = [-80, -44, -22, 10]
shp_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\ThreeRegions_Boundary\clip\Amazon_ThreeRegions_Clip.shp'

cls_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig1\Data\Forest_Mask\MCD12Q1_Amazon.tif' #_500m
_, _, cls_md = readTif_gdal_safe(cls_path)
cls_md_forest = cls_md.copy()
cls_md_forest[cls_md_forest != 2] = -1
cls_modis_coarse = cv2.resize(cls_md_forest, (raws_y, columns_x))
forest_mask = (cls_modis_coarse == 2)

sif_path_1 = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\SIF_Benchmark\Amazon_GOSIF_mean_calibrated.tif'
sif_path_2 = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\SIF_Benchmark\CSIF_Amazon_005_calibrated.tif'
_, _, sif_1 = readTif_gdal_safe(sif_path_1)
_, _, sif_2 = readTif_gdal_safe(sif_path_2)

sif_1 = cv2.resize(sif_1, (raws_y, columns_x))
sif_2 = cv2.resize(sif_2, (raws_y, columns_x))

sif1_ts = sif_1[forest_mask, :] * 0.0001

sif2_ts = sif_2[forest_mask, :]
sif1_ts[sif1_ts < 0] = np.nan
sif2_ts[sif2_ts < 0] = np.nan
sif_ref_nor = np.nanmean(np.stack([normalize_ts(sif1_ts), normalize_ts(sif2_ts)], axis=0), axis=0)

# sif_ref_nor = np.nanmean([sif1_ts, sif2_ts], axis=0)
# sif_ref_nor = normalize_ts(sif1_ts)

# In[]

print(">>> 正在加载 Exp1 与 Exp3 进行对比计算 ")

exp1_dir = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\GPP\Multi_Experiments\Exp1_satellite_input'
exp3_dir = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\GPP\Multi_Experiments\Exp3_dec_LAI_Demography_input'

model_pairs = {
    'EC-LUE': (['EC_LUE_EVI_GPP_local.tif', 'EC_LUE_kNDVI_GPP_local.tif', 'EC_LUE_NDVI_GPP_local.tif',
                'EC_LUE_MODIS_LAI_GPP_local.tif'],
               'EC_LUE_LAI_Dec_Demography_GPP.tif'),
    'MODIS-LUE': (['MOD_LUE_GPP_local.tif'],
                'MOD_LUE_LAI_Dec_Demography_GPP.tif'),
    'TL-EC': (['TL_EC_LUE_GPP_local.tif'],
              'TL_EC_LUE_LAI_Dec_Demography_GPP.tif')
}

delta_r_maps = {}
formulation_mean_delta_r_list = []  # 用于最终计算 Ensemble Mean

for model_name, (exp1_files, file_exp3) in model_pairs.items():
    # 1. 读取该 Formulation 对应的 Exp3 (LD model)
    exp3_path = os.path.join(exp3_dir, file_exp3)
    if not os.path.exists(exp3_path):
        print(f"❌ 找不到 Exp3 文件: {file_exp3}，跳过该模型。")
        continue

    geo_info, _, data_exp3 = readTif_gdal_safe(exp3_path)
    data_exp3 = cv2.resize(data_exp3, (raws_y, columns_x))
    ts_exp3 = data_exp3[forest_mask, :]

    specific_delta_r_maps = []  # 存储当前 formulation 下每一个 specific model 的 delta r 空间图

    # 2. 遍历该 Formulation 下的每一个 Exp1 specific model
    for f1 in exp1_files:
        f_path = os.path.join(exp1_dir, f1)
        if not os.path.exists(f_path):
            print(f"⚠️ 找不到文件: {f1}，跳过该文件。")
            continue

        _, _, data_exp1 = readTif_gdal_safe(f_path)
        data_exp1 = cv2.resize(data_exp1, (raws_y, columns_x))
        ts_exp1 = data_exp1[forest_mask, :]

        # 3. 针对当前的 specific pair 构建联合掩膜 (至少 6 个月有效)
        valid_mask_ind = (np.sum(~np.isnan(sif_ref_nor), axis=1) >= 6) & \
                         (np.sum(~np.isnan(ts_exp1), axis=1) >= 6) & \
                         (np.sum(~np.isnan(ts_exp3), axis=1) >= 6)

        # 4. 计算相关的 r 值
        r_exp1 = vectorized_pearsonr(ts_exp1[valid_mask_ind], sif_ref_nor[valid_mask_ind])
        r_exp3 = vectorized_pearsonr(ts_exp3[valid_mask_ind], sif_ref_nor[valid_mask_ind])

        # 5. 求出该 specific model 的 delta r
        delta_r_ind = r_exp3 - r_exp1

        # 6. 将当前 specific model 的 delta r 映射回 2D 空间图
        map_delta_r_ind = np.full(forest_mask.shape, np.nan)
        temp_r_ind = np.full(forest_mask.sum(), np.nan)
        temp_r_ind[valid_mask_ind] = delta_r_ind
        map_delta_r_ind[forest_mask] = temp_r_ind

        specific_delta_r_maps.append(map_delta_r_ind)

    if specific_delta_r_maps:
        # 7. 计算当前 Formulation 的 Mean Delta r
        # 使用 np.nanmean 忽略缺失像元，保证鲁棒性
        mean_formulation_delta_r = np.nanmean(np.stack(specific_delta_r_maps, axis=0), axis=0)

        # 存入字典和系综列表
        delta_r_maps[model_name] = mean_formulation_delta_r
        formulation_mean_delta_r_list.append(mean_formulation_delta_r)

        print(f"✅ 完成对比: {model_name} (综合了 {len(specific_delta_r_maps)} 个 specific model 的 Delta r)")
    else:
        print(f"⚠️ {model_name} 没有有效的 specific models 参与计算。")

# 8. 计算最终的系综总体平均改进 (Ensemble Mean)
print(">>> 正在计算系综总体平均改进 (Ensemble Mean of Mean Delta r)...")
if formulation_mean_delta_r_list:
    # 针对 3 个 formulation 的 mean delta r 求大均值
    ensemble_mean_delta_r = np.nanmean(np.stack(formulation_mean_delta_r_list, axis=0), axis=0)
    delta_r_maps['Ensemble Mean'] = ensemble_mean_delta_r
    print("✅ 完成 Ensemble Mean 计算。")
else:
    print("❌ 没有足够的 formulation 数据来计算 Ensemble Mean。")

# In[] average seasonality and then evaluate performance -- slightly worse

# print(">>> 正在加载 Exp1 与 Exp3 进行对比计算 (Ensemble GPP -> Delta R Method)...")
#
# exp1_dir = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\GPP\Multi_Experiments\Exp1_satellite_input'
# exp3_dir = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\VPM_Model\GPP\Multi_Experiments\Exp3_dec_LAI_Demography_input'
#
# model_pairs = {
#     'EC-LUE': (['EC_LUE_EVI_GPP_local.tif', 'EC_LUE_kNDVI_GPP_local.tif', 'EC_LUE_NDVI_GPP_local.tif',
#                 'EC_LUE_MODIS_LAI_GPP_local.tif'],
#                'EC_LUE_LAI_Dec_Demography_GPP.tif'),
#     'MOD-LUE': (['MOD_LUE_GPP_local.tif'],
#                 'MOD_LUE_LAI_Dec_Demography_GPP.tif'),
#     'TL-EC': (['TL_EC_LUE_GPP_local.tif'],
#               'TL_EC_LUE_LAI_Dec_Demography_GPP.tif')
# }
#
# delta_r_maps = {}
# formulation_mean_delta_r_list = []  # 用于最终计算 3 大框架的大系综
#
# for model_name, (exp1_files, file_exp3) in model_pairs.items():
#     # ---------------------------------------------------------
#     # 1. 读取该 Formulation 对应的 Exp3 (LD model)
#     # ---------------------------------------------------------
#     exp3_path = os.path.join(exp3_dir, file_exp3)
#     if not os.path.exists(exp3_path):
#         print(f"❌ 找不到 Exp3 文件: {file_exp3}，跳过该模型。")
#         continue
#
#     geo_info, _, data_exp3 = readTif_gdal_safe(exp3_path)
#     data_exp3 = cv2.resize(data_exp3, (raws_y, columns_x))
#     ts_exp3 = data_exp3[forest_mask, :]  # Shape: (pixels, 12)
#
#     # ---------------------------------------------------------
#     # 2. 读取并平均该 Formulation 下所有的 Exp1 specific models
#     # ---------------------------------------------------------
#     tmp_exp1_ts_list = []
#
#     for f1 in exp1_files:
#         f_path = os.path.join(exp1_dir, f1)
#         if not os.path.exists(f_path):
#             print(f"⚠️ 找不到文件: {f1}，跳过该文件。")
#             continue
#
#         _, _, data_exp1 = readTif_gdal_safe(f_path)
#         data_exp1 = cv2.resize(data_exp1, (raws_y, columns_x))
#         tmp_exp1_ts_list.append(data_exp1[forest_mask, :])
#
#     if not tmp_exp1_ts_list:
#         print(f"⚠️ {model_name} 没有有效的 Exp1 输入，跳过计算。")
#         continue
#
#     # 【核心逻辑变更】：在计算 R 之前，先将多个 Exp1 模型的时间序列进行融合平均
#     # 堆叠后 shape: (num_models, pixels, 12) -> 求均值后 shape: (pixels, 12)
#     ts_exp1_mean = np.nanmean(np.stack(tmp_exp1_ts_list, axis=0), axis=0)
#
#     # ---------------------------------------------------------
#     # 3. 联合掩膜与相关系数计算
#     # ---------------------------------------------------------
#     # 确保三个序列（观测、融合Baseline、LD模型）在同一点都有至少6个月的有效值
#     valid_mask_ind = (np.sum(~np.isnan(sif_ref_nor), axis=1) >= 6) & \
#                      (np.sum(~np.isnan(ts_exp1_mean), axis=1) >= 6) & \
#                      (np.sum(~np.isnan(ts_exp3), axis=1) >= 6)
#
#     # 计算相关的 Pearson r 值
#     r_exp1_mean = vectorized_pearsonr(ts_exp1_mean[valid_mask_ind], sif_ref_nor[valid_mask_ind])
#     r_exp3 = vectorized_pearsonr(ts_exp3[valid_mask_ind], sif_ref_nor[valid_mask_ind])
#
#     # 求出该 formulation 下的 delta r
#     delta_r_ind = r_exp3 - r_exp1_mean
#
#     # ---------------------------------------------------------
#     # 4. 映射回空间矩阵并存储
#     # ---------------------------------------------------------
#     map_delta_r_ind = np.full(forest_mask.shape, np.nan)
#     temp_r_ind = np.full(forest_mask.sum(), np.nan)
#     temp_r_ind[valid_mask_ind] = delta_r_ind
#     map_delta_r_ind[forest_mask] = temp_r_ind
#
#     delta_r_maps[model_name] = map_delta_r_ind
#     formulation_mean_delta_r_list.append(map_delta_r_ind)
#
#     print(f"✅ 完成对比: {model_name} (先融合了 {len(tmp_exp1_ts_list)} 个 Exp1 序列，再计算 Delta r)")
#
# # ---------------------------------------------------------
# # 5. 计算最终的三大框架总体平均改进 (Ensemble Mean)
# # ---------------------------------------------------------
# print(">>> 正在计算三大框架总体平均改进 (Ensemble Mean of the 3 Frameworks)...")
# if formulation_mean_delta_r_list:
#     # 针对 3 个 formulation (EC-LUE, MOD-LUE, TL-EC) 求大均值
#     ensemble_mean_delta_r = np.nanmean(np.stack(formulation_mean_delta_r_list, axis=0), axis=0)
#     delta_r_maps['Ensemble Mean'] = ensemble_mean_delta_r
#     print("✅ 完成 Ensemble Mean 计算。")
# else:
#     print("❌ 没有足够的 formulation 数据来计算 Ensemble Mean。")

# In[]
print(">>> 正在提取站点尺度时间序列并计算 ±1 SEM 阴影带...")

eddy_flux_name_list = ['K34_CfluxBF_DOY.csv', 'K67_CfluxBF_DOY.csv', 'CAX_CfluxBF_DOY.csv', 'RJA_CfluxBF_DOY.csv']
site_locate_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\Evaluation\eddy_fluxes\Eddy_Fluxes_Sites_Locations.xlsx'
eddy_flux_dir = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Fig3\Data\Evaluation\eddy_fluxes\DOY'
site_locate_pd = pd.read_excel(site_locate_path)

site_series_data = {}
r_bench_all, rmse_bench_all = [], []
r_ens_all, rmse_ens_all = [], []
EC_pos = []

# ==============================================================================
# 步骤 1: 将之前读取的所有 Exp1 和 Exp3 的 3D 矩阵存入列表，以备站点提取
# （注意：假设你前一步读取时，已经有读取数据的手段，这里我们直接构建两个池子）
# ==============================================================================
# 这些列表将直接存储 shape 为 (row, col, 12) 的原始矩阵
raw_exp1_matrices = []
raw_exp3_matrices = []
raw_exp1_labels = []
raw_exp3_labels = []

for model_name, (exp1_files, file_exp3) in model_pairs.items():
    formulation_exp1_matrices = []
    # 提取所有 Exp1 矩阵
    for f1 in exp1_files:
        f_path = os.path.join(exp1_dir, f1)
        if os.path.exists(f_path):
            _, _, data = readTif_gdal_safe(f_path)
            formulation_exp1_matrices.append(cv2.resize(data, (raws_y, columns_x)))

    if len(formulation_exp1_matrices) > 0:
        raw_exp1_matrices.append(np.nanmean(np.stack(formulation_exp1_matrices, axis=0), axis=0))
        raw_exp1_labels.append(model_name)

    # 提取对应的 Exp3 矩阵
    exp3_path = os.path.join(exp3_dir, file_exp3)
    if os.path.exists(exp3_path):
        _, _, data_exp3 = readTif_gdal_safe(exp3_path)
        raw_exp3_matrices.append(cv2.resize(data_exp3, (raws_y, columns_x)))
        raw_exp3_labels.append(model_name)

print(
    f"📦 Site extraction uses formulation-balanced Exp1 models: {raw_exp1_labels}; "
    f"LD(Exp3) models: {raw_exp3_labels}.")

# ==============================================================================
# 步骤 2: 循环提取每个站点的时序与验证
# ==============================================================================

for eddy_flux_name in eddy_flux_name_list:
    site_name = eddy_flux_name.split('_')[0]
    site_info = site_locate_pd[site_locate_pd['Site'] == site_name]
    lat, lon = site_info['Latitude'].values[0], site_info['Longitude'].values[0]
    EC_pos.append((lat, lon))

    # 坐标转换
    col = int((lon - geo_info[0]) / geo_info[1])
    row = int((lat - geo_info[3]) / geo_info[5])


    r_start, r_end = max(0, row - 1), row + 1
    c_start, c_end = max(0, col - 1), col + 1

    # --- 提取 LD Models (Exp3) 的时序 ---
    # shape 期望: (num_models, 12)
    site_exp3_models = np.array([
        np.nanmean(d[r_start:r_end, c_start:c_end, :], axis=(0, 1))
        for d in raw_exp3_matrices
    ])
    site_exp3_mean = np.nanmean(site_exp3_models, axis=0)
    # 注意: ddof=1 用于计算无偏样本标准差
    site_exp3_sem = np.nanstd(site_exp3_models, axis=0, ddof=1) / np.sqrt(site_exp3_models.shape[0])

    # --- 提取 Baseline Models (Exp1) 的时序 ---
    site_exp1_models = np.array([
        np.nanmean(d[r_start:r_end, c_start:c_end, :], axis=(0, 1))
        for d in raw_exp1_matrices
    ])
    site_exp1_mean = np.nanmean(site_exp1_models, axis=0)
    site_exp1_sem = np.nanstd(site_exp1_models, axis=0, ddof=1) / np.sqrt(site_exp1_models.shape[0])

    # --- 读取观测数据 (Obs) ---
    csv_path = os.path.join(eddy_flux_dir, eddy_flux_name)
    pd_eddy = pd.read_csv(csv_path, usecols=['DOY', 'GEP', 'Month'])
    pd_eddy['GEP'] = pd.to_numeric(pd_eddy['GEP'], errors='coerce')
    site_obs = pd_eddy.groupby('Month')['GEP'].mean().to_numpy()

    # --- 指标计算 (r 和 RMSE) ---
    v_mask = ~np.isnan(site_obs) & ~np.isnan(site_exp3_mean) & ~np.isnan(site_exp1_mean)

    if np.sum(v_mask) > 3:  # 确保有足够有效月
        # LD model (Exp3) 评估
        r_exp3 = np.corrcoef(site_exp3_mean[v_mask], site_obs[v_mask])[0, 1]
        rmse_exp3 = np.sqrt(np.mean((site_exp3_mean[v_mask] - site_obs[v_mask]) ** 2))
        r_bench_all.append(r_exp3)
        rmse_bench_all.append(rmse_exp3)

        # Baseline model (Exp1) 评估
        r_exp1 = np.corrcoef(site_exp1_mean[v_mask], site_obs[v_mask])[0, 1]
        rmse_exp1 = np.sqrt(np.mean((site_exp1_mean[v_mask] - site_obs[v_mask]) ** 2))
        r_ens_all.append(r_exp1)
        rmse_ens_all.append(rmse_exp1)
    else:
        print(f"⚠️ {site_name} 站点有效月份不足，无法计算 R 和 RMSE。")

    # 存储用于画折线图和阴影带的数据
    site_series_data[site_name] = {
        'obs': site_obs,
        'exp3_m': site_exp3_mean, 'exp3_sem': site_exp3_sem,
        'exp1_m': site_exp1_mean, 'exp1_sem': site_exp1_sem
    }

print("✅ 所有站点时间序列及误差带提取完毕！")


# In[] ==========================================
# 5. 输出引擎 A：生成主图 (1 Map + 站点验证)
# ==========================================
# cmap_blue = plt.cm.Blues_r(np.linspace(0.20, 0.80, 6))
# cmap_red = plt.cm.Reds(np.linspace(0.15, 0.80, 6))
# colors = np.vstack((cmap_blue, cmap_red))
# custom_cmap = mcolors.ListedColormap(colors)
# levels = [-0.8, -0.60, -0.4, -0.2, 0, 0.2, 0.40, 0.6, 0.8]
# norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=12)

cmap_blue = plt.cm.Blues_r(np.linspace(0.10, 0.80, 5))
cmap_red = plt.cm.Reds(np.linspace(0.10, 0.80, 5))
colors = np.vstack((cmap_blue, cmap_red))
custom_cmap = mcolors.ListedColormap(colors)

# 2. 设置 11 个边界，恰好切分出 10 个色块
# 最外层的 [-1.0, -0.8] 和 [0.8, 1.0] 就是用来装载 ≥0.8 和 ≤-0.8 的极端值的
levels = [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=10)
print(">>> 正在生成正文主图 (Main Figure 4)...")
import string

fig_main = plt.figure(figsize=(15.5,8), dpi=200)
proj = ccrs.PlateCarree()

plt.rcParams['font.family'] = 'Helvetica'

# ==========================================
# 面板 (a): 地图
# ==========================================
ax_map = fig_main.add_axes([0.05, 0.20, 0.46, 0.72], projection=proj)  # 略微调整左边距给标号留空间
# ax_map.add_feature(cfeature.LAND, facecolor='#EBEBEB', edgecolor='none', zorder=1)
# ax_map.add_feature(cfeature.OCEAN, facecolor='#FFFFFF', edgecolor='none', zorder=2, alpha=0.8)
ax_map.add_feature(cfeature.LAND, facecolor='white', zorder=1)
ax_map.add_feature(cfeature.OCEAN, zorder=2)
# ax_map.add_feature(cfeature.COASTLINE, linewidth=0.4, color='#999999', zorder=3)
add_shp(ax_map, shp_path, lw=0.8, ec='k', fc='none', zorder=4)

map_data_ens = delta_r_maps['Ensemble Mean']
im_main = ax_map.imshow(map_data_ens, origin='upper', extent=img_extent, transform=proj,
                        zorder=3, cmap=custom_cmap, norm=norm)
ax_map.set_extent(map_extent, crs=proj)
ax_map.set_xticks(np.arange(-80, -40, 10), crs=proj)
ax_map.set_yticks(np.arange(-20, 20, 10), crs=proj)
ax_map.xaxis.set_major_formatter(LongitudeFormatter())
ax_map.yaxis.set_major_formatter(LatitudeFormatter())

# 【修改 1】：刻度向外 (direction='out')
ax_map.tick_params(axis='both', labelsize=16, colors='k', length=4, width=0.8, direction='out',
                   left=True, right=False, top=True, bottom=False,
                   labelright=False, labeltop=True, labelbottom=False)

# 【修改 2】：移除 title 中的标号，使用 text 独立定位在左上角外侧
# ax_map.set_title('Improvement: Ensemble Mean (Exp3 vs Exp1)', fontweight='bold', fontsize=16, pad=15)
# ax_map.text(-0.05, 1.05, 'a', transform=ax_map.transAxes, fontsize=18, fontweight='bold', va='bottom')

# for i, pos in enumerate(EC_pos):
#     ax_map.plot(pos[1], pos[0], marker='*', color='k', ms=20, mec='white', zorder=5)
#     ax_map.text(pos[1] - 0.6, pos[0] - 2.5, eddy_flux_name_list[i].split('_')[0], color="k", fontsize=14,
#                 fontweight='bold', zorder=6)

cbar_ax = fig_main.add_axes([0.10, 0.11, 0.36, 0.025])  # 配合地图左移微调
cb_main = fig_main.colorbar(im_main, cax=cbar_ax, orientation='horizontal')
cb_main.set_ticks([-0.8, -0.4, 0, 0.4, 0.8])

# 5. 设置标签 (使用 LaTeX 语法输出极其优雅的 ≤ 和 ≥)
cb_main.set_ticklabels(['≤ -0.8', '-0.4', '0', '0.4', r'0.8 ≤'], fontsize=14)
cb_main.set_label(r'$\Delta r$', fontsize=16, fontweight='bold')

# 细节微调：让刻度线向内 (in)，模仿参考图的样式，并设置线宽
cb_main.ax.tick_params(direction='in', length=5, width=1.2, colors='k')
cb_main.outline.set_linewidth(1.2) # 外边框加粗

# --- 嵌入饼状图 ---
# valid_data = map_data_ens[~np.isnan(map_data_ens)]
# if len(valid_data) > 0:
#     improved_cnt = np.sum(valid_data >= 0)
#     degraded_cnt = np.sum(valid_data < 0)
#     sizes = [improved_cnt, degraded_cnt]
#     pie_colors_sig = ['#cb181d', '#2166ac']
#
#     ax_ins = ax_map.inset_axes([0.70, 0.01, 0.34, 0.34], zorder=10)  # 稍微左移一点防止挤出边界
#     wedges, texts, autotexts = ax_ins.pie(
#         sizes, colors=pie_colors_sig, autopct='%1.1f%%', startangle=90, pctdistance=0.65,
#         wedgeprops=dict(edgecolor=None, linewidth=0.5, alpha=0.9, width=0.65)
#     )
#
#     for j, autotext in enumerate(autotexts):
#         autotext.set_fontsize(14)
#         autotext.set_fontweight('bold')
#         autotext.set_color('white')
#         if sizes[j] / sum(sizes) < 0.05:
#             autotext.set_text('')

# --- 嵌入 Delta r 频率直方图 ---
valid_data = map_data_ens[np.isfinite(map_data_ens)]
percent_improved_gpp = np.mean(valid_data > 0)

# 计算 Δr 分布的中位数和四分位距
q1_delta_r, median_delta_r, q3_delta_r = np.percentile(
    valid_data, [25, 50, 75]
)
iqr_delta_r = q3_delta_r - q1_delta_r

print(
    'The distribution was centered on modest positive changes, '
    f'with a median Δr of {median_delta_r:.3f} '
    f'(IQR: {q1_delta_r:.3f}–{q3_delta_r:.3f}).'
)
print(f'IQR width (Q3 − Q1): {iqr_delta_r:.3f}')

# 直方图在地图坐标轴中的位置
left, bottom, width, height = 0.70, 0.005, 0.26, 0.235

# 白色底板的范围略大于直方图，遮住其后方的地图和黑色边界。
# zorder=8：高于地图和边界；直方图使用 zorder=10，保证不会被底板遮挡。
from matplotlib.patches import Rectangle

board_pad_left = 0.025
board_pad_right = 0.065
board_pad_bottom = 0.065
board_pad_top = 0.025

histogram_board = Rectangle(
    (left - board_pad_left, bottom - board_pad_bottom),
    width + board_pad_left + board_pad_right,
    height + board_pad_bottom + board_pad_top,
    transform=ax_map.transAxes,
    facecolor='white',
    edgecolor='none',
    linewidth=0,
    zorder=8,
    clip_on=False
)
ax_map.add_patch(histogram_board)

# 将直方图绘制在白色底板上方
ax1 = ax_map.inset_axes(
    [left, bottom, width, height],
    transform=ax_map.transAxes,
    zorder=10
)
ax1.set_facecolor('white')
ax1.patch.set_alpha(1.0)

sns.histplot(
    data=valid_data,
    stat='probability',
    bins=50,
    color='gray',
    alpha=0.75,
    edgecolor='black',
    linewidth=0.25,
    ax=ax1,
    zorder=11
)

ax1.yaxis.set_ticks_position('right')
ax1.yaxis.set_label_position('right')
ax1.set_xlim(-1.0, 1.0)
ax1.set_ylim(0, 0.18)
ax1.set_ylabel('Density', fontsize=12, labelpad=2)
ax1.set_xlabel(r'$\Delta r$', fontsize=12, labelpad=1)

ax1.axvline(
    0,
    color='darkred',
    linewidth=1.0,
    linestyle='--',
    zorder=12
)
ax1.text(
    0.505,
    0.94,
    fr'$\Delta r$ > 0: {percent_improved_gpp * 100:.1f}%',
    transform=ax1.transAxes,
    fontsize=10,
    color='darkred',
    ha='left',
    va='top',
    zorder=13
)

ax1.tick_params(
    axis='both',
    which='both',
    colors='black',
    labelsize=10,
    direction='out',
    length=3,
    width=0.7
)

# 保留直方图自身的细边框，但后方地图边界会被白色底板完全遮住
for spine in ax1.spines.values():
    spine.set_color('black')
    spine.set_linewidth(0.7)



# ==========================================
# 面板 (b)-(e): 站点折线图
# ==========================================
line_coords = [[0.57, 0.71, 0.18, 0.22], [0.80, 0.71, 0.18, 0.22], [0.57, 0.395, 0.18, 0.22], [0.80, 0.395, 0.18, 0.22]]
m_list = np.arange(1, 13)
line_handles = []

for i, site in enumerate([n.split('_')[0] for n in eddy_flux_name_list]):
    ax = fig_main.add_axes(line_coords[i])
    d = site_series_data[site]

    ax.fill_between(m_list, d['exp3_m'] - d['exp3_sem'], d['exp3_m'] + d['exp3_sem'], color=COLOR_EXP3, alpha=0.20,
                    lw=0, zorder=2)
    l1 = ax.plot(m_list, d['exp3_m'], color=COLOR_EXP3, ls='-', lw=2, marker='^', ms=7, mfc=COLOR_EXP3, mec='white',
                 zorder=5)

    ax.fill_between(m_list, d['exp1_m'] - d['exp1_sem'], d['exp1_m'] + d['exp1_sem'], color=COLOR_EXP1, alpha=0.20,
                    lw=0, zorder=1)
    l2 = ax.plot(m_list, d['exp1_m'], color=COLOR_EXP1, ls='-', lw=2, marker='s', ms=6, mfc=COLOR_EXP1, mec='white',
                 zorder=4)

    l3 = ax.plot(m_list, d['obs'], color=COLOR_OBS, ls='--', lw=1.5, marker='o', ms=6, mfc='white', mec=COLOR_OBS,
                 zorder=6)
    if i == 0: line_handles = [l1[0], l2[0], l3[0]]

    # 【修改 1】：刻度向外
    ax.tick_params(axis='both', direction='out', labelsize=13)

    # 【修改 2】：移除 title 中的标号，独立定位于左上角
    ax.set_title(f'{site}', fontsize=15, fontweight='bold', pad=8)
    # 使用 string.ascii_lowercase 获取 b, c, d, e
    letter = string.ascii_lowercase[i + 1]
    ax.text(0.00, 1.05, letter, transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom')

    if i % 2 == 0: ax.set_ylabel(r'GPP $(\mathrm{g\,C\,m^{-2}\,d^{-1}})$', fontsize=15)
    ax.set_xticks(m_list)
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize=13)
    ax.set_ylim(4.5, 11.0 if site != 'RJA' else 11.5)
    ax.set_xlim(0.5, 12.5)

fig_main.legend(handles=line_handles, labels=['LD-LUE ± 1 SEM', 'Conv-LUE ± 1 SEM', 'EC-derived GPP'],
                loc='center', bbox_to_anchor=(0.775, 0.325), ncol=3, frameon=False, fontsize=13)

# ==========================================
# 面板 (f) & (g): 柱状图
# ==========================================
bar_y = np.arange(4)
width = 0.35

# --- 面板 f ---
ax_r = fig_main.add_axes([0.57, 0.12, 0.18, 0.15])  # 稍微压缩高度给标题留空间
b1 = ax_r.barh(bar_y - width / 2, r_bench_all[::-1], width, color=COLOR_EXP3, zorder=3)
b2 = ax_r.barh(bar_y + width / 2, r_ens_all[::-1], width, color=COLOR_EXP1, zorder=3)

# 【修改 1】：刻度向外
ax_r.tick_params(axis='both', direction='out', labelsize=13)

# 【修改 2】：不再将 (f) 作为 title，而是独立标注
ax_r.set_title('', loc='left')  # 清空原 title
ax_r.text(0.00, 1.05, 'f', transform=ax_r.transAxes, fontsize=18, fontweight='bold', va='bottom')

ax_r.set_xlabel('Correlation coefficient ($r$)', fontsize=15)
ax_r.set_yticks(bar_y)
ax_r.set_yticklabels([n.split('_')[0] for n in eddy_flux_name_list][::-1])

# --- 面板 g ---
ax_rmse = fig_main.add_axes([0.80, 0.12, 0.18, 0.15])
ax_rmse.barh(bar_y - width / 2, rmse_bench_all[::-1], width, color=COLOR_EXP3, zorder=3)
ax_rmse.barh(bar_y + width / 2, rmse_ens_all[::-1], width, color=COLOR_EXP1, zorder=3)

# 【修改 1】：刻度向外
ax_rmse.tick_params(axis='both', direction='out', labelsize=13)

# 【修改 2】：不再将 (g) 作为 title，而是独立标注
ax_rmse.set_title('', loc='left')
ax_rmse.text(0.00, 1.05, 'g', transform=ax_rmse.transAxes, fontsize=18, fontweight='bold', va='bottom')

ax_rmse.set_xlabel(r'RMSE $(\mathrm{g\,C\,m^{-2}\,d^{-1}})$', fontsize=15)
ax_rmse.set_yticks(bar_y)
ax_rmse.set_yticklabels([n.split('_')[0] for n in eddy_flux_name_list][::-1])

fig_main.legend(handles=[b1, b2], labels=['LD-LUE', 'Conv-LUE'],
                loc='center', bbox_to_anchor=(0.775, 0.025), ncol=2, frameon=False, fontsize=15)

plt.show()

# save_path = r'J:\PhD_Works\Work4_Amazon_Pattern_Detection_ECAE\Main_Figures\Nature_Revision_Round1\FigS_LD_LUE_Conv_LUE_Comparison.png'
# fig_main.savefig(save_path, dpi=300, bbox_inches='tight')

# In[] ==========================================
# 6. 输出引擎 B：生成附图 (1x3 个体模型评估矩阵)
# ==========================================
print(">>> 正在生成 Nature Supplementary 评估矩阵 (1x3)...")
rows, cols = 1, 3
fig_sup = plt.figure(figsize=(12, 5.5), dpi=300)

# 使用 GridSpec 控制精准布局，抛弃 tight_layout
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(rows, cols, figure=fig_sup, wspace=0.25)

plot_order = ['EC-LUE', 'TL-EC', 'MODIS-LUE']
plot_tile = ['EC-LUE', 'TL-EC-LUE', 'MODIS-LUE']

panel_letters = ['a', 'b', 'c']

# 饼图的专属颜色：红 (Improved), 蓝 (Degraded), 灰 (Neutral)
# 对应色带提取经典色彩
# pie_colors = ['#cb181d', '#2166ac', '#e0e0e0']

for i, model_name in enumerate(plot_order):
    ax = fig_sup.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
    map_data = delta_r_maps[model_name]  # 请确保您的 delta_r_maps 已定义

    # ==========================================
    # 🌟 核心优化 1：顶级期刊底图配色
    # 陆地使用极干净的浅灰，海洋纯白，加入极细的海岸线
    # ==========================================
    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.OCEAN,  edgecolor='none', zorder=2, alpha=0.8)
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color='#999999', zorder=3)

    add_shp(ax, shp_path, lw=0.8, ec='k', fc='none', zorder=5)

    im = ax.imshow(map_data, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(),
                   zorder=4, cmap=custom_cmap, norm=norm)

    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(-80, -40, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-20, 20, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # 仅最左侧显示 Y 轴标签，所有图显示 X 轴标签
    show_left_label = (i == 0)
    ax.tick_params(axis='both', labelsize=14, colors='k', length=4, width=0.8,
                   left=True, right=False, top=True, bottom=False,
                   labelleft=show_left_label, labelright=False, labeltop=True, labelbottom=False)

    # ==========================================
    # 🌟 核心优化 2：绝对对齐的字母标号与标题
    # ==========================================
    model_name = plot_tile[i]
    ax.set_title(f'{model_name}', fontsize=16, pad=12, y=1.10)
    ax.text(-0.1, 1.18, panel_letters[i], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom')

    # ==========================================
    # 嵌入 Delta r 频率直方图
    # ==========================================
    valid_data = map_data[np.isfinite(map_data)]

    if len(valid_data) > 0:
        percent_improved = np.mean(valid_data > 0)

        # 直方图在地图坐标轴中的位置
        left, bottom, width, height = 0.70, 0.02, 0.27, 0.23

        # 白色底板，遮住底图和边界线，保证直方图清晰
        from matplotlib.patches import Rectangle

        board_pad_left = 0.025
        board_pad_right = 0.060
        board_pad_bottom = 0.060
        board_pad_top = 0.025

        histogram_board = Rectangle(
            (left - board_pad_left, bottom - board_pad_bottom),
            width + board_pad_left + board_pad_right,
            height + board_pad_bottom + board_pad_top,
            transform=ax.transAxes,
            facecolor='white',
            edgecolor='none',
            linewidth=0,
            zorder=8,
            clip_on=False
        )
        ax.add_patch(histogram_board)

        # 嵌入直方图
        ax_ins = ax.inset_axes(
            [left, bottom, width, height],
            transform=ax.transAxes,
            zorder=10
        )
        ax_ins.set_facecolor('white')
        ax_ins.patch.set_alpha(1.0)

        sns.histplot(
            data=valid_data,
            stat='probability',
            bins=50,
            color='gray',
            alpha=0.75,
            edgecolor='black',
            linewidth=0.25,
            ax=ax_ins,
            zorder=11
        )

        ax_ins.axvline(
            0,
            color='darkred',
            linewidth=1.0,
            linestyle='--',
            zorder=12
        )

        ax_ins.text(
            0.50,
            1.18,
            fr'$\Delta r$ > 0: {percent_improved * 100:.1f}%',
            transform=ax_ins.transAxes,
            fontsize=8.5,
            color='darkred',
            ha='left',
            va='top',
            zorder=13
        )

        ax_ins.yaxis.set_ticks_position('right')
        ax_ins.yaxis.set_label_position('right')

        ax_ins.set_xlim(-1.0, 1.0)
        ax_ins.set_ylim(0, 0.18)

        ax_ins.set_xlabel(r'$\Delta r$', fontsize=9, labelpad=1)
        ax_ins.set_ylabel('Density', fontsize=9, labelpad=1)

        ax_ins.tick_params(
            axis='both',
            which='both',
            colors='black',
            labelsize=8,
            direction='out',
            length=2.5,
            width=0.6
        )

        for spine in ax_ins.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.6)
        # 调整中心文字的字号以匹配加厚的色环
        # ax_ins.text(0, 0, 'Sig.\nOnly', ha='center', va='center',
        #             fontsize=6, fontweight='bold', color='#333333')
# ==========================================
# 🌟 核心优化 4：全局 Colorbar 居中放置
# ==========================================
# 调整画布整体边距，底部留出空间
fig_sup.subplots_adjust(left=0.06, right=0.96, top=0.85, bottom=0.18)

# 在底部中央添加细长优雅的 Colorbar
# cbar_ax = fig_sup.add_axes([0.25, 0.12, 0.5, 0.025])
# cb_main = fig_sup.colorbar(im, cax=cbar_ax, orientation='horizontal')
# cb_main.ax.xaxis.set_ticks([-1.00, -0.50, -neutral_th, neutral_th, 0.50, 1.00])
# cb_main.ax.xaxis.set_ticklabels(['≤ -1.0', '-0.5', f'-{neutral_th}', f'{neutral_th}', '0.5', '1.0 ≤'])

cbar_ax = fig_sup.add_axes([0.25, 0.12, 0.5, 0.025])
cb_main = fig_sup.colorbar(im, cax=cbar_ax, orientation='horizontal')
cb_main.ax.xaxis.set_ticks(np.array([-0.80, -0.40, 0, 0.40, 0.80]))

# 5. 设置标签 (使用 LaTeX 语法输出极其优雅的 ≤ 和 ≥)
cb_main.set_ticklabels(['≤ -0.8', '-0.4', '0', '0.4', r'0.8 ≤'],fontsize=14)
cb_main.set_label(r'$\Delta r$', fontsize=16, fontweight='bold')


# cb_main.set_label(r'$\Delta r$', fontsize=13, fontweight='bold', labelpad=6)
# cb_main.ax.tick_params(labelsize=11, length=3)

# 外边框线也调细
cb_main.outline.set_linewidth(0.8)

plt.show()
print(f"\n>>> 全部图表绘制完成！")
#
# save_path_sup = r'D:\OneDrive - The University Of Hong Kong\PhD_Projects\Project4_Mapping_Amazon_Basin_Using_GEE\Manuscript\coauthor\Song et al Nature\Revision_round3\Revised_Figures\FigS_Three_Framework_LD_LUE_Conv_LUE_Comparison_Supplement.png'
# fig_sup.savefig(save_path_sup, dpi=300, bbox_inches='tight')
