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

import multiprocessing as mul
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
import tracemalloc
from scipy.optimize import minimize
import numba
# from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
# 设置环境变量防止多线程冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
mul.set_start_method('spawn', force=True)

matplotlib.use('Qt5Agg')
matplotlib.rcParams['figure.dpi'] = 150



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


def load_and_preprocess_data(file_path):
    """加载并预处理TIFF数据"""
    _geo, _prj, composite_data = readTif_gdal(file_path)
    composite_data = composite_data.astype(np.float32)
    composite_data[composite_data >= 1000] = np.nan

    return _geo, _prj, composite_data/1000.0 # normalize to [0, 1] dec fraction


def reshape_and_mask_data(composite_data):
    """数据重塑和掩码处理"""
    # 重塑为(12, N)的形状
    composite_data_res = composite_data.reshape(-1, 12).transpose(1, 0)

    # 创建有效数据掩码 if contain NAN, then remove
    valid_mask = np.all(~np.isnan(composite_data_res), axis=0)

    return composite_data_res, valid_mask


# Numba加速的叶龄动态模型核心计算
# ================== 核心算法部分 ==================
@numba.jit(nopython=True)
def leaf_age_dynamic_model_numba(LAIy0, LAIm0, LAI_total, Flush, Ty_to_M, TM_to_O):
    """叶龄动态模型核心计算（Numba加速）"""
    months = len(LAI_total)
    LAIY = np.zeros(months)
    LAIM = np.zeros(months)
    LAIO = np.zeros(months)
    simulated_litterfall = np.zeros(months)

    LAIY[0] = LAIy0
    LAIM[0] = LAIm0
    LAIO[0] = max(LAI_total[0] - LAIy0 - LAIm0, 1e-3)

    inv_Ty = 1.0 / Ty_to_M
    inv_TM = 1.0 / TM_to_O

    for t in range(months - 1):
        dLAIy = Flush[t] - LAIY[t] * inv_Ty
        dLAIm = LAIY[t] * inv_Ty - LAIM[t] * inv_TM

        LAIY_next = LAIY[t] + dLAIy
        LAIM_next = LAIM[t] + dLAIm
        LAIO_next = LAI_total[t + 1] - LAIY_next - LAIM_next

        if LAIO_next < 1e-3:
            preserved_LAIO = LAIO[t] * 0.5 * inv_TM
            excess = (LAIY_next + LAIM_next + preserved_LAIO) - LAI_total[t + 1]

            if excess > 0:
                total_YM = LAIY_next + LAIM_next
                scale = (LAI_total[t + 1] - preserved_LAIO) / total_YM if total_YM > 0 else 0.0
                LAIY_next *= scale
                LAIM_next *= scale

            LAIO_next = preserved_LAIO

        LAIY[t + 1] = max(LAIY_next, 0.0)
        LAIM[t + 1] = max(LAIM_next, 0.0)
        LAIO[t + 1] = max(LAIO_next, 0.0)

        simulated_litterfall[t] = LAIO[t] + LAIM[t] * inv_TM - LAIO[t + 1]

    half = months // 2
    return LAIY[half:], LAIM[half:], LAIO[half:], simulated_litterfall[half:]


def leaf_demography_model(dec_data_site, LAI_total_max=6, Ty_to_M=1, TM_to_O=3):
    """叶龄结构建模与参数优化"""
    # LAI_litterfall = LAI_total_max * dec_data_site
    LAI_litterfall = 4.79* dec_data_site + 0.58 # based on the relationship between litterfall LAI and deciduousness

    # LAI = LAI_total_max * (1 - dec_data_site) #LAI=8.241*leaf canopy fraction-1.991
    LAI = 8.241 * (1 - dec_data_site) - 1.991

    delta_LAI = np.empty_like(LAI)
    delta_LAI[:-1] = np.diff(LAI)
    delta_LAI[-1] = LAI[0] - LAI[-1]
    flush_data = np.maximum(delta_LAI + LAI_litterfall, 0)

    LAI_ext = np.concatenate([LAI, LAI])
    flush_ext = np.concatenate([flush_data, flush_data])

    def error_minimize(LAIy0, LAIm0):
        if (LAIy0 + LAIm0 > LAI_total_max) or (LAIy0 < 0) or (LAIm0 < 0):
            return -1e6
        try:
            *_, simulated = leaf_age_dynamic_model_numba(
                LAIy0, LAIm0, LAI_ext, flush_ext, Ty_to_M, TM_to_O
            )
            rmse = np.sqrt(np.mean((simulated - LAI_litterfall) ** 2))
            corr = np.corrcoef(simulated, LAI_litterfall)[0, 1]
            return -rmse + 2 * corr
        except:
            return -1e6

    optimizer = BayesianOptimization(
        f=error_minimize,
        pbounds={
            "LAIy0": (0.1, min(5, LAI_total_max - 0.1)),
            "LAIm0": (0.1, min(5, LAI_total_max - 0.1))
        },
        random_state=42,
        allow_duplicate_points=True,
        verbose=0
    )

    optimizer.set_gp_params(
        kernel=Matern(nu=2.5),
        alpha=1e-3,
        n_restarts_optimizer=3  # 减少优化次数
    )

    optimizer.maximize(
        init_points=3,  # 减少初始点
        n_iter=15,  # 减少迭代次数
        acquisition_function=UtilityFunction(kind="ei", xi=0.2)
    )

    params = optimizer.max['params']
    LAIY, LAIM, LAIO, _ = leaf_age_dynamic_model_numba(
        params['LAIy0'], params['LAIm0'], LAI_ext, flush_ext, Ty_to_M, TM_to_O
    )

    return LAIY, LAIM, LAIO


def process_pixel_column(args):
    """处理单个像素列，返回叶龄数据"""
    i, dec_col = args
    try:
        LAIY, LAIM, LAIO = leaf_demography_model(dec_col)
        # 返回三个叶龄阶段的数组堆叠
        return i, np.vstack([LAIY, LAIM, LAIO])
    except Exception as e:
        return i, np.zeros((3, len(dec_col))) * np.nan
# ================== 并行处理部分 ==================
def main_samp_extract(pars):
    """主处理函数（单进程处理分块数据）"""
    shm_EVI_name, shm_EVI_out_name = pars[0], pars[1]
    shape_EVI, shape_EVI_out = pars[2], pars[3]
    dtype_EVI, dtype_EVI_out = pars[4], pars[5]
    st, ed = pars[6], pars[7]

    try:
        shm_EVI = SharedMemory(name=shm_EVI_name, create=False)
        shm_EVI_out = SharedMemory(name=shm_EVI_out_name, create=False)

        EVI_ts = np.ndarray(shape_EVI, dtype_EVI, buffer=shm_EVI.buf)
        EVI_out_ts = np.ndarray(shape_EVI_out, dtype_EVI_out, buffer=shm_EVI_out.buf)

        # 逐列处理避免内部并行
        for i in tqdm(range(st, ed)):

            if i >= EVI_ts.shape[1]:
                break
            try:
                LAIY, LAIM, LAIO = leaf_demography_model(EVI_ts[:, i])
                # 存储三个叶龄阶段
                EVI_out_ts[0, :, i] = LAIY
                EVI_out_ts[1, :, i] = LAIM
                EVI_out_ts[2, :, i] = LAIO
            except:
                EVI_out_ts[:, :, i] = np.nan

        return 1

    finally:
        shm_EVI.close()
        shm_EVI_out.close()



# In[]
# ================== 主程序入口 ==================
if __name__ == '__main__':
    tracemalloc.start()
    start_time = time.time()

    # 数据加载和预处理
    tif_path = r'..\Deciduousness_Seasonality_Amazon.tif' # real deciduousenss seasonality data

    _geo, _prj, composite_data = load_and_preprocess_data(tif_path)
    original_shape = composite_data.shape

    # 数据重塑和掩码处理
    composite_data_res, valid_mask = reshape_and_mask_data(composite_data)
    # 数据重塑和掩码处理
    EVI_ts = composite_data_res[:, valid_mask]
    EVI_out_ts = np.zeros_like(EVI_ts)

    EVI_out_ts = np.zeros((3, *EVI_ts.shape)) * np.nan  # [3, 12, N]


    # 共享内存处理
    with SharedMemoryManager() as smm:
        # 创建共享内存
        shm_EVI = smm.SharedMemory(size=EVI_ts.nbytes)
        shm_EVI_out = smm.SharedMemory(size=EVI_out_ts.nbytes)

        # 数据拷贝到共享内存
        np.copyto(np.ndarray(EVI_ts.shape, EVI_ts.dtype, buffer=shm_EVI.buf), EVI_ts)
        np.copyto(np.ndarray(EVI_out_ts.shape, EVI_out_ts.dtype, buffer=shm_EVI_out.buf), EVI_out_ts)

        # 分块参数配置
        CONFIG = {
            "chunk_size": 1000,  # 每个分块处理1000个像素
            "max_workers": 48      # 并行进程数
        }
        parameters = [
            (
                shm_EVI.name,
                shm_EVI_out.name,
                EVI_ts.shape,
                EVI_out_ts.shape,
                EVI_ts.dtype,
                EVI_out_ts.dtype,
                st,
                min(st + CONFIG['chunk_size'], EVI_ts.shape[1])
            )
            for st in range(0, EVI_ts.shape[1], CONFIG['chunk_size'])
        ]

        # 多进程处理
        with mul.Pool(processes=CONFIG['max_workers']) as pool:
            results = []
            for _ in pool.imap(main_samp_extract, parameters):
                results.append(_)

        composite_data_res = np.zeros((3, *composite_data_res.shape))  # [3, 12, H*W]
        composite_data_res[:, :, valid_mask] = np.ndarray(
            EVI_out_ts.shape,
            EVI_out_ts.dtype,
            buffer=shm_EVI_out.buf
        )

    # 结果保存
    save_dir = r'..\SI_Leaf_Demography_Model'

    os.makedirs(save_dir, exist_ok=True)

    # 转换为空间格式 [H, W, 36]
    data_final = composite_data_res.transpose(2, 1, 0).reshape(
        composite_data.shape[0],
        composite_data.shape[1],
        3 * 12
    )
    # 保存36波段的叶龄数据
    save_tif(data_final.astype(np.float32),
             os.path.join(save_dir, 'Leaf_Age_Wu_2016.tif'),
             _geo, _prj, 36)

    # save_tif(data_final.astype(np.float32), os.path.join(save_dir, 'LA_LUE_GPP.tif'), _geo, _prj, 12)

    # 性能报告
    print(f'Total time: {time.time() - start_time:.2f}s')
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory usage - Current: {current / 1e6}MB, Peak: {peak / 1e6}MB")
    tracemalloc.stop()
