# -*- coding: utf-8 -*-
"""
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Function:
        1.read and save the Tif data
        2. Cloud removal algorithm in Wang et al. 2021
%       Copyright @ Guangqin Song at HKU, 2025
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
# In[]
# import the required packages
import random
import numpy as np
from VCA import vca
import matplotlib.pyplot as plt
import copy
import unmixing_V2
import cv2
import os
from osgeo import gdal, gdalconst
from skimage import morphology
import warnings
from skimage.segmentation import felzenszwalb

try:
    import gdal
except:
    from osgeo import gdal

warnings.filterwarnings('ignore')


# In[]

def truncated_linear_stretch(image, truncated_value, max_out=255, min_out=0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        if (max_out <= 255):
            gray = np.uint8(gray)
        elif (max_out <= 65535):
            gray = np.uint16(gray)
        return gray

    #  如果是多波段
    if (len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
    #  如果是单波段
    else:
        image_stretch = gray_process(image)
    return image_stretch


# read the tif format data (i.e. multi-spectral satellite data)
def readTif_gdal(fileName,nbands=20):
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
            # im_data = np.transpose(im_data, [1, 2, 0])
    return dataset.GetGeoTransform(), dataset.GetProjection(), im_data


# save the tif format data with the original coordinate (i.e. multi-spectral satellite data)
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
    else:
        for i in range(nbands):
            outputData.GetRasterBand(i + 1).WriteArray(grouthTif[:, :, i])
    del outputData


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
    if nbands == 1:
        outputData.GetRasterBand(1).WriteArray(grouthTif)
        outputData.GetRasterBand(1).SetNoDataValue(9999)
        outputData.GetRasterBand(1).SetNoDataValue(65535)
    else:
        for i in range(nbands):
            outputData.GetRasterBand(i + 1).WriteArray(grouthTif[:, :, i])
            outputData.GetRasterBand(i + 1).SetNoDataValue(9999)
            outputData.GetRasterBand(1).SetNoDataValue(65535)

    del outputData


def gasuss_noise(image, mean=0, var=0.005, proportion=0.4):  # default 0.001
    img_max = image.max()
    img_nor = np.array(image / img_max, dtype=float)
    # img_nor = np.array(image, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img_nor.shape)
    random.seed(42)
    index_all = img_nor.shape[0]
    index = random.sample(range(index_all), int(index_all * proportion))
    gaussian_out = copy.deepcopy(img_nor)
    gaussian_out[index] = img_nor[index] + noise[index]
    # gaussian_out = img_nor + noise
    gaussian_out = gaussian_out * img_max
    gaussian_out = np.clip(gaussian_out, 0, 1)
    return gaussian_out


# obtain the NDVI map
def get_NDVI_map(img):
    nir = img[:, :, 3]
    red = img[:, :, 2]
    ndvi_map = np.divide((nir - red), (nir + red), out=np.zeros_like((nir - red), dtype=np.float64),
                         where=(nir + red) != 0)
    return ndvi_map


# obtain the endmember from the whole image
def get_abundance_map_V2(img, end_num=3, file_name=None):
    ndvi_map = get_NDVI_map(img)
    new_img = copy.copy(img)
    # remove the non-vegetation
    # new_img[np.where(ndvi_map < 0.4)] = 0
    new_img = np.nan_to_num(new_img)
    index = 0
    end_record = 0
    height, width, bands = img.shape

    # It is difficult to extract endmember from the whole image
    # extract the endmember
    input_vca = new_img.reshape(height * width, bands).transpose([1, 0])
    mask = (input_vca == 0).all(0)
    input_vca = input_vca[:, ~mask]
    # obtain the endmember from the vca
    Ed_mean, ind, _ = vca(input_vca, 3, verbose=False, snr_input=30)

    # reorder the endmember according the Nir value
    Ed_mean = Ed_mean[:, Ed_mean[3, :].argsort()]  # the second dimension is the deciduousness

    if file_name:
        plt.figure()
        plt.plot(Ed_mean[:, 0], color='k', label='shade')
        plt.plot(Ed_mean[:, 1], color='r', label='NPV')
        plt.plot(Ed_mean[:, 2], color='g', label='GV')
        plt.legend()
        plt.title(file_name)

    input_img = new_img.reshape(height * width, bands).transpose([1, 0])
    # using the linear spectral unmixing to obtain the abundance map
    img_abundance = unmixing_V2.sunsal(Ed_mean, input_img, al_iters=200, positivity=True, addone=True, verbose=False)[
        0].transpose(
        [1, 0])
    img_abundance = img_abundance.reshape(height, width, end_num)

    return img_abundance


def resampling(source_file, target_file, scale=1.0):
    """
    影像重采样
    :param source_file: 源文件
    :param target_file: 输出影像
    :param scale: 像元缩放比例
    :return:
    """
    dataset = gdal.Open(source_file, gdalconst.GA_ReadOnly)
    band_count = dataset.RasterCount  # 波段数

    # if band_count == 0 or not scale > 0:
    #     print("参数异常")
    #     return
    cols = dataset.RasterXSize  # 列数
    rows = dataset.RasterYSize  # 行数
    cols = int(cols * scale)  # 计算新的行列数
    rows = int(rows * scale)

    geotrans = list(dataset.GetGeoTransform())

    print(dataset.GetGeoTransform())
    # print(geotrans)
    geotrans[1] = geotrans[1] / scale  # 像元宽度变为原来的scale倍
    geotrans[5] = geotrans[5] / scale  # 像元高度变为原来的scale倍
    print(geotrans)

    if os.path.exists(target_file) and os.path.isfile(target_file):  # 如果已存在同名影像
        os.remove(target_file)  # 则删除之

    band1 = dataset.GetRasterBand(1)
    data_type = band1.DataType
    target = dataset.GetDriver().Create(target_file, xsize=cols, ysize=rows, bands=band_count,
                                        eType=data_type)
    target.SetProjection(dataset.GetProjection())  # 设置投影坐标
    target.SetGeoTransform(geotrans)  # 设置地理变换参数

    # gdal.ReprojectImage(dataset, target, dataset.GetProjection(), dataset.GetProjection(), gdalconst.GRA_Bilinear)

    total = band_count + 1
    for index in range(1, total):
        # 读取波段数据
        print("正在写入" + str(index) + "波段")
        data = dataset.GetRasterBand(index).ReadAsArray(buf_xsize=cols, buf_ysize=rows,
                                                        resample_alg=gdalconst.GRA_Average)  # GRA_Bilinear

        out_band = target.GetRasterBand(index)
        # out_band.SetNoDataValue(dataset.GetRasterBand(index).GetNoDataValue())
        out_band.WriteArray(data)  # 写入数据到新影像中
        out_band.FlushCache()  # 关闭文件
        out_band.ComputeBandStats(False)  # 计算统计信息

    print("正在写入完成")
    del dataset
    del target


# In[] 腐蚀 erode testing

def erode(data):
    result_data = np.zeros_like(data)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for i in range(data.shape[2]):
        erode_data = cv2.erode(data[:, :, i], kernel)
        result_data[:, :, i] = erode_data

    mask = np.all(result_data.reshape(-1, data.shape[2]) == 0, axis=1)
    img_new = data.reshape(-1, data.shape[2])
    img_new[mask] = 0
    img_new = img_new.reshape(data.shape)
    return img_new


# In[]
# import cv2
# cv2.dilate()
# 开运算
def open(image, win=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    dst = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return dst


# 闭操作
def close(image, win=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    dst = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return dst


def morphlg(img, win=7):  # 形态学运算剔除噪声点

    # img_open = open(img.astype(np.uint8), win)
    # img_close = close(img.astype(np.uint8), win)
    img_bwopen = morphology.remove_small_objects(img.astype(bool), min_size=250, connectivity=1)
    # mis_pixmph = close(img_bwopen.astype(np.uint8), win)
    # mis_pixmph = open(img_bwopen.astype(np.uint8), win)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    img_dilate = cv2.dilate(img_bwopen.astype(np.uint8), kernel)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # mis_pixmph = cv2.dilate(mis_pixmph, kernel)
    return img_dilate



def cloud_removal_qc(data_dir, save_dir, threshold=[5, 95], data_type='S2'):

    img_list = [x for x in os.listdir(data_dir) if x.endswith('.tif')]
    Geo_, Prj_, img_tmp = readTif_gdal(os.path.join(data_dir, img_list[0]))
    h, w, nbands = img_tmp.shape
    nbands = 4
    img_all = np.zeros([h, w, nbands, len(img_list)])
    Ndvi_all = np.zeros([h, w, len(img_list)])
    NIRv_all = np.zeros([h, w, len(img_list)])
    CIred_all = np.zeros([h, w, len(img_list)])

    for tmp_i in range(len(img_list)):
        img_name = img_list[tmp_i]
        img_path = os.path.join(data_dir, img_name)
        # read the planet image
        Geo_, Prj_, img_tmp = readTif_gdal(img_path)
        # img_tmp = img_tmp
        if data_type == 'S2':

            img_tmp = img_tmp[:, :, [0, 1, 2, 6]]
            mask_zero = img_tmp.sum(axis=2) == 0
            img_tmp[mask_zero] = np.nan
            ndvi_tmp = get_NDVI_map(img_tmp)
            nir = img_tmp[:, :, 3]
            red = img_tmp[:, :, 2]

            CIred = nir / red - 1

            img_tmp[ndvi_tmp < 0.2] = np.nan

            img_all[:, :, :, tmp_i] = img_tmp
            # img_all[tmp_i, :, :, :] = img_tmp[:, :, :]
            Ndvi_all[:, :, tmp_i] = get_NDVI_map(img_tmp)
            NIRv_all[:, :, tmp_i] = ndvi_tmp * nir
            CIred_all[:, :, tmp_i] = CIred

        elif data_type == 'Landsat':

            img_tmp = img_tmp[:, :, [0, 1, 2, 3]]
            img_all[:, :, :, tmp_i] = img_tmp  # [:, :, [0, 1, 2, 3]]
            # img_all[tmp_i, :, :, :] = img_tmp[:, :, :]
            Ndvi_all[:, :, tmp_i] = get_NDVI_map(img_tmp[:, :, [0, 1, 2, 3]])

            # mask_zero = img_tmp.sum(axis=2) == 0
            # img_tmp[mask_zero] = 0

            ndvi_tmp = get_NDVI_map(img_tmp)
            nir = img_tmp[:, :, 3]
            red = img_tmp[:, :, 2]

            CIred = nir / red - 1

            img_tmp[ndvi_tmp < 0.2] = 0

            img_all[:, :, :, tmp_i] = img_tmp
            # img_all[tmp_i, :, :, :] = img_tmp[:, :, :]
            Ndvi_all[:, :, tmp_i] = get_NDVI_map(img_tmp)
            NIRv_all[:, :, tmp_i] = ndvi_tmp * nir
            CIred_all[:, :, tmp_i] = CIred

    img_all = img_all.astype(np.float)
    img_all[img_all == 0] = np.nan
    # centralization
    img_new = img_all

    img_new = img_new - np.nanmean(img_new, axis=(0, 1),
                                   keepdims=True)  # + np.nanmean(img_new, axis=(0, 1, 3),keepdims=True)

    prc_threshold_10 = np.nanpercentile(img_new, threshold[0], axis=(0, 1, 3), keepdims=True)
    prc_threshold_90 = np.nanpercentile(img_new, threshold[1], axis=(0, 1, 3), keepdims=True)

    # step 1 find out the shade region
    NIRv_mask_all = NIRv_all < 0.18

    planet_outlier_v1 = np.sum((img_new < prc_threshold_10), axis=2) > 0  # | (img_new > prc_threshold_90)
    # planet_outlier_v1 = np.logical_and(planet_outlier_v1,NIRv_mask_all)

    # step 2 find out the cloud region while exclude vegetation
    Ndvi_mask_all_v2 = Ndvi_all < 0.75  # aviod removing the vegetation part

    CIred_mask = CIred_all < 6  # sensitivity to cloud

    cloud_mask_all = np.sum((img_new > prc_threshold_90), axis=2) > 0

    planet_outlier_v2 = cloud_mask_all  # np.logical_and(cloud_mask_all, Ndvi_mask_all_v2)

    # final mask
    planet_outlier = np.logical_or(planet_outlier_v2, planet_outlier_v1)

    # planet_outlier = np.sum(planet_outlier, axis=3) > 1
    for tmp_i in range(len(img_list)):
        # img_name = img_list[tmp_i]
        img_name = img_list[tmp_i]
        img_path = os.path.join(data_dir, img_name)
        # read the planet image
        Geo_, Prj_, img_tmp = readTif_gdal(img_path)

        # remove water in BCI
        if data_type == 'S2':
            img_fourbands = img_tmp[:, :, [0, 1, 2, 6]]
        else:
            img_fourbands = img_tmp[:, :, [0, 1, 2, 3]]
        ndvi_tmp = get_NDVI_map(img_fourbands)

        green = img_fourbands[:, :, 1]
        nir = img_fourbands[:, :, 3]
        ndwi = np.divide((green - nir), (green + nir), where=(green + nir) != 0)

        img_tmp[ndvi_tmp < 0.2] = 0

        # img_tmp[ndwi >= -0.35] = 0

        mask_raw = img_tmp.sum(axis=2) == 0
        mask_raw = mask_raw.astype(bool)

        planet_mis = planet_outlier[:, :, tmp_i]

        mis_pixmph = planet_mis.astype(bool)

        mis_pixmph = np.logical_or(mask_raw, mis_pixmph)

        # print(np.sum(mis_pixmph)/mis_pixmph.reshape(-1).shape[0])
        if np.sum(mis_pixmph) / mis_pixmph.reshape(-1).shape[0] > 0.7:
            continue

        mis_pixmph = morphology.remove_small_objects(mis_pixmph, min_size=64, connectivity=1,
                                                     )  # obtain the cloud mask

        win = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
        img_dilate = cv2.dilate(mis_pixmph.astype(np.uint8), kernel)

        mis_pixmph = img_dilate.astype(bool)

        # cloud removal
        img_tmp[mis_pixmph, :] = 0

        # remove the samll object from the raw image
        mask = img_tmp.sum(axis=2) != 0

        mask_remove = morphology.remove_small_objects(mask, min_size=256, connectivity=1)
        img_tmp[~mask_remove, :] = 0

        save_tif(img_tmp, os.path.join(save_dir, img_name.replace('.tif', '_qc.tif')), Geo_, Prj_, img_tmp.shape[2])

        del img_tmp, mask, mask_remove, planet_mis, mis_pixmph, img_dilate

def cloud_removal_qc_gee(img_list, threshold=None, data_type='S2'):
    if threshold is None:
        threshold = [10, 90]

    img_tmp = img_list[0]
    h, w, nbands = img_tmp.shape
    nbands = 4
    img_all = np.zeros([h, w, nbands, len(img_list)])
    Ndvi_all = np.zeros([h, w, len(img_list)])
    img_all_qc = np.zeros_like(img_all)

    for tmp_i in range(len(img_list)):
        img_tmp = img_list[tmp_i]
        # img_tmp = img_tmp
        if data_type == 'S2':
            img_all[:, :, :, tmp_i] = img_tmp
            # img_all[tmp_i, :, :, :] = img_tmp[:, :, :]
            Ndvi_all[:, :, tmp_i] = get_NDVI_map(img_tmp)
        elif data_type == 'Landsat':
            img_all[:, :, :, tmp_i] = img_tmp[:, :, [0, 1, 2, 3]]
            # img_all[tmp_i, :, :, :] = img_tmp[:, :, :]
            Ndvi_all[:, :, tmp_i] = get_NDVI_map(img_tmp[:, :, [0, 1, 2, 3]])

    img_all = img_all.astype(np.float)
    # img_all[img_all == 0] = np.nan
    # centralization
    img_new = img_all
    img_new = img_new - np.nanmean(img_new, axis=(0, 1), keepdims=True) + np.nanmean(img_new, axis=(0, 1, 3),
                                                                                     keepdims=True)

    prc_threshold_10 = np.nanpercentile(img_new, threshold[0], axis=(0, 1, 3), keepdims=True)
    prc_threshold_90 = np.nanpercentile(img_new, threshold[1], axis=(0, 1, 3), keepdims=True)

    # step 1 find out the shade region
    planet_outlier_v1 = np.sum((img_new < prc_threshold_10), axis=2) > 1  # | (img_new > prc_threshold_90)
    # step 2 find out the cloud region while exclude vegetation
    Ndvi_mask_all = Ndvi_all < 0.8
    cloud_mask_all = (np.sum((img_new > prc_threshold_90), axis=2) > 1)
    planet_outlier_v2 = np.logical_and(cloud_mask_all, Ndvi_mask_all)
    planet_outlier = np.logical_or(planet_outlier_v2, planet_outlier_v1)

    for tmp_i in range(len(img_list)):
        # img_name = img_list[tmp_i]
        planet_mis = planet_outlier[:, :, tmp_i]

        mis_pixmph = planet_mis.astype(bool)
        mis_pixmph = morphology.remove_small_objects(mis_pixmph, min_size=64, connectivity=1,
                                                     in_place=False)  # obtain the cloud mask
        win = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
        img_dilate = cv2.dilate(mis_pixmph.astype(np.uint8), kernel)

        mis_pixmph = img_dilate.astype(bool)

        # mis_pixmph = mis_pixmph & ndvi_map
        img_tmp = img_list[tmp_i].astype(np.float16)

        img_tmp[mis_pixmph, :] = np.nan

        # remove the samll object from the raw image
        mask = np.isnan(img_tmp).sum(axis=2) == 0

        mask_remove = morphology.remove_small_objects(mask, min_size=64, connectivity=1, in_place=False)
        img_tmp[~mask_remove, :] = np.nan

        img_all_qc[:, :, :, tmp_i] = img_tmp

        del img_tmp, mask, mask_remove, planet_mis, mis_pixmph, img_dilate

    return img_all_qc
