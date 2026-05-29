"""Remote-sensing preprocessing helpers.

"""

# In[] Imports
import random

import numpy as np

from vca import vca
import matplotlib

import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import unmixing
import scipy.io as scio
import scipy.io as sio
import cv2
import os
from sklearn.cluster import KMeans
import math
from osgeo import gdal, gdalconst
from skimage import morphology
import warnings
from skimage.segmentation import felzenszwalb

# In[] Workflow
warnings.filterwarnings('ignore')

# In[] Functions
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

    if (len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)

    else:
        image_stretch = gray_process(image)
    return image_stretch

def readTif_gdal(fileName, nbands=10):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print("cannot open file:" + fileName)
        return
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    if im_data.shape[0] <= nbands:
        im_data = np.transpose(im_data, [1, 2, 0])
    return dataset.GetGeoTransform(), dataset.GetProjection(), im_data

def save_tif(grouthTif, savePath, Geo_, Projection_, nbands):
    gdal.UseExceptions()
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    datatype = gdal.GDT_Float32

    outputData = driver.Create(savePath, grouthTif.shape[1], grouthTif.shape[0], nbands, datatype)

    if (outputData != None):
        outputData.SetGeoTransform(Geo_)
        outputData.SetProjection(Projection_)

    if nbands == 1:
        outputData.GetRasterBand(1).WriteArray(grouthTif)
        outputData.GetRasterBand(1).SetNoDataValue(np.nan)
    else:
        for i in range(nbands):
            outputData.GetRasterBand(i + 1).WriteArray(grouthTif[:, :, i])
            outputData.GetRasterBand(i + 1).SetNoDataValue(np.nan)
    del outputData

def save_tif_int(grouthTif, savePath, Geo_, Projection_, nbands):

    driver = gdal.GetDriverByName("GTiff")
    driver.Register()

    datatype = gdal.GDT_UInt16

    outputData = driver.Create(savePath, grouthTif.shape[1], grouthTif.shape[0], nbands, datatype,
                               options=["COMPRESS=LZW", 'TILED=YES', 'PREDICTOR=2'])

    outputData.SetGeoTransform(Geo_)
    outputData.SetProjection(Projection_)

    if nbands == 1:
        outputData.GetRasterBand(1).WriteArray(grouthTif)
        outputData.GetRasterBand(1).SetNoDataValue(9999)
    else:
        for i in range(nbands):
            outputData.GetRasterBand(i + 1).WriteArray(grouthTif[:, :, i])
            outputData.GetRasterBand(i + 1).SetNoDataValue(9999)

    del outputData

def gaussian_noise(image, mean=0, var=0.005, proportion=0.4):
    img_max = image.max()
    img_nor = np.array(image / img_max, dtype=float)

    noise = np.random.normal(mean, var ** 0.5, img_nor.shape)
    random.seed(42)
    index_all = img_nor.shape[0]
    index = random.sample(range(index_all), int(index_all * proportion))
    gaussian_out = copy.deepcopy(img_nor)
    gaussian_out[index] = img_nor[index] + noise[index]

    gaussian_out = gaussian_out * img_max
    gaussian_out = np.clip(gaussian_out, 0, 1)
    return gaussian_out

def get_NDVI_map(img):
    nir = img[:, :, 3]
    red = img[:, :, 2]
    ndvi_map = np.divide((nir - red), (nir + red), out=np.zeros_like((nir - red), dtype=np.float64),
                         where=(nir + red) != 0)
    return ndvi_map

def get_abundance_map_V2(img, end_num=3, file_name=None):
    ndvi_map = get_NDVI_map(img)
    new_img = copy.copy(img)

    new_img = np.nan_to_num(new_img)
    index = 0
    end_record = 0
    height, width, bands = img.shape

    input_vca = new_img.reshape(height * width, bands).transpose([1, 0])
    mask = (input_vca == 0).all(0)
    input_vca = input_vca[:, ~mask]

    Ed_mean, ind, _ = vca(input_vca, 3, verbose=False, snr_input=30)
    '''
    patch_size = 20
    index = 0
    end_record = 0
    for h in range(height // patch_size):
        for w in range(width // patch_size):
            img_cut = new_img[h * patch_size: (h + 1) * patch_size, w * patch_size: (w + 1) * patch_size, :]
            img_cut = img_cut.reshape(patch_size * patch_size, bands)
            img_cut = img_cut.transpose([1, 0])
            if np.sum(img_cut) == 0:
                continue
            else:

                mask = (img_cut == 0).all(0)
                column_indices = np.where(mask)[0]
                input_vca = img_cut[:, ~mask]

                Ed_sub, ind, _ = vca(input_vca, 3, verbose=False, snr_input=30)

                end_record = end_record + 1
                if index == 0:
                    Ed_all = Ed_sub
                    Ed_all_m = Ed_sub
                    Img_list = input_vca
                else:
                    Ed_all_m = np.concatenate((Ed_all_m, Ed_sub), axis=1)
                    Ed_all = Ed_all + Ed_sub

            index = index + 1

    kmeans = KMeans(n_clusters=end_num, random_state=0, n_init=10).fit(Ed_all_m.transpose([1, 0]))
    kmeans_centers = kmeans.cluster_centers_
    Ed_mean = kmeans_centers.transpose([1,0])
    '''

    Ed_mean = Ed_mean[:, Ed_mean[3, :].argsort()]

    if file_name:
        plt.figure()
        plt.plot(Ed_mean[:, 0], color='k', label='shade')
        plt.plot(Ed_mean[:, 1], color='r', label='NPV')
        plt.plot(Ed_mean[:, 2], color='g', label='GV')
        plt.legend()
        plt.title(file_name)

    input_img = new_img.reshape(height * width, bands).transpose([1, 0])

    img_abundance = unmixing.sunsal(Ed_mean, input_img, al_iters=200, positivity=True, addone=True, verbose=False)[
        0].transpose(
        [1, 0])
    img_abundance = img_abundance.reshape(height, width, end_num)

    return img_abundance

def resampling(source_file, target_file, scale=1.0):
    dataset = gdal.Open(source_file, gdalconst.GA_ReadOnly)
    band_count = dataset.RasterCount

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    cols = int(cols * scale)
    rows = int(rows * scale)

    geotrans = list(dataset.GetGeoTransform())

    print(dataset.GetGeoTransform())

    geotrans[1] = geotrans[1] / scale
    geotrans[5] = geotrans[5] / scale
    print(geotrans)

    if os.path.exists(target_file) and os.path.isfile(target_file):
        os.remove(target_file)

    band1 = dataset.GetRasterBand(1)
    data_type = band1.DataType
    target = dataset.GetDriver().Create(target_file, xsize=cols, ysize=rows, bands=band_count,
                                        eType=data_type)
    target.SetProjection(dataset.GetProjection())
    target.SetGeoTransform(geotrans)

    total = band_count + 1
    for index in range(1, total):

        print("Writing band " + str(index))
        data = dataset.GetRasterBand(index).ReadAsArray(buf_xsize=cols, buf_ysize=rows,
                                                        resample_alg=gdalconst.GRA_Average)

        out_band = target.GetRasterBand(index)

        out_band.WriteArray(data)
        out_band.FlushCache()
        out_band.ComputeBandStats(False)

    print("Finished writing raster.")
    del dataset
    del target

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

def open(image, win=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    dst = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return dst

def close(image, win=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    dst = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return dst

def morphlg(img, win=7):

    img_bwopen = morphology.remove_small_objects(img.astype(bool), min_size=250, connectivity=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    img_dilate = cv2.dilate(img_bwopen.astype(np.uint8), kernel)

    return img_bwopen

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

        Geo_, Prj_, img_tmp = readTif_gdal(img_path)

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

            Ndvi_all[:, :, tmp_i] = get_NDVI_map(img_tmp)
            NIRv_all[:, :, tmp_i] = ndvi_tmp * nir
            CIred_all[:, :, tmp_i] = CIred

        elif data_type == 'Landsat':

            img_tmp = img_tmp[:, :, [0, 1, 2, 3]]
            img_all[:, :, :, tmp_i] = img_tmp

            Ndvi_all[:, :, tmp_i] = get_NDVI_map(img_tmp[:, :, [0, 1, 2, 3]])

            ndvi_tmp = get_NDVI_map(img_tmp)
            nir = img_tmp[:, :, 3]
            red = img_tmp[:, :, 2]

            CIred = nir / red - 1

            img_tmp[ndvi_tmp < 0.2] = 0

            img_all[:, :, :, tmp_i] = img_tmp

            Ndvi_all[:, :, tmp_i] = get_NDVI_map(img_tmp)
            NIRv_all[:, :, tmp_i] = ndvi_tmp * nir
            CIred_all[:, :, tmp_i] = CIred

    img_all = img_all.astype(np.float)
    img_all[img_all == 0] = np.nan

    img_new = img_all

    img_new = img_new - np.nanmean(img_new, axis=(0, 1),
                                   keepdims=True)

    prc_threshold_10 = np.nanpercentile(img_new, threshold[0], axis=(0, 1, 3), keepdims=True)
    prc_threshold_90 = np.nanpercentile(img_new, threshold[1], axis=(0, 1, 3), keepdims=True)

    NIRv_mask_all = NIRv_all < 0.18

    planet_outlier_v1 = np.sum((img_new < prc_threshold_10), axis=2) > 0

    Ndvi_mask_all_v2 = Ndvi_all < 0.75

    CIred_mask = CIred_all < 6

    cloud_mask_all = np.sum((img_new > prc_threshold_90), axis=2) > 0

    planet_outlier_v2 = cloud_mask_all

    planet_outlier = np.logical_or(planet_outlier_v2, planet_outlier_v1)

    for tmp_i in range(len(img_list)):

        img_name = img_list[tmp_i]
        img_path = os.path.join(data_dir, img_name)

        Geo_, Prj_, img_tmp = readTif_gdal(img_path)

        if data_type == 'S2':
            img_fourbands = img_tmp[:, :, [0, 1, 2, 6]]
        else:
            img_fourbands = img_tmp[:, :, [0, 1, 2, 3]]
        ndvi_tmp = get_NDVI_map(img_fourbands)

        green = img_fourbands[:, :, 1]
        nir = img_fourbands[:, :, 3]
        ndwi = np.divide((green - nir), (green + nir), where=(green + nir) != 0)

        img_tmp[ndvi_tmp < 0.2] = 0

        mask_raw = img_tmp.sum(axis=2) == 0
        mask_raw = mask_raw.astype(bool)

        planet_mis = planet_outlier[:, :, tmp_i]

        mis_pixmph = planet_mis.astype(bool)

        mis_pixmph = np.logical_or(mask_raw, mis_pixmph)

        mis_pixmph = morphology.remove_small_objects(mis_pixmph, min_size=64, connectivity=1,
                                                     in_place=False)

        win = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
        img_dilate = cv2.dilate(mis_pixmph.astype(np.uint8), kernel)

        mis_pixmph = img_dilate.astype(bool)

        img_tmp[mis_pixmph, :] = 0

        mask = img_tmp.sum(axis=2) != 0

        mask_remove = morphology.remove_small_objects(mask, min_size=256, connectivity=1, in_place=False)
        img_tmp[~mask_remove, :] = 0

        save_tif(img_tmp, os.path.join(save_dir, img_name.replace('.tif', '_qc.tif')), Geo_, Prj_, img_tmp.shape[2])

        del img_tmp, mask, mask_remove, planet_mis, mis_pixmph, img_dilate

def cloud_removal_qc_superpixel(data_dir, save_dir, threshold=[5, 95], data_type='S2'):

    img_list = [x for x in os.listdir(data_dir) if x.endswith('.tif')]
    Geo_, Prj_, img_first = readTif_gdal(os.path.join(data_dir, img_list[0]))

    h, w, nbands = img_first.shape
    nbands = 4
    img_all = np.zeros([h, w, nbands, len(img_list)])
    Ndvi_all = np.zeros([h, w, len(img_list)])
    NIRv_all = np.zeros([h, w, len(img_list)])
    CIred_all = np.zeros([h, w, len(img_list)])
    Seg_cloud_mask_all = np.zeros([h, w, len(img_list)])

    for tmp_i in range(len(img_list)):

        img_name = img_list[tmp_i]
        img_path = os.path.join(data_dir, img_name)

        Geo_, Prj_, img_tmp = readTif_gdal(img_path)

        if not img_tmp.shape == (h, w, nbands):
            img_tmp = cv2.resize(img_tmp, (w, h))

        if data_type == 'S2':
            img_tmp = img_tmp[:, :, [0, 1, 2, 6]]
        elif data_type == 'Landsat':
            img_tmp = img_tmp[:, :, [0, 1, 2, 3]]

        img_tmp = np.nan_to_num(img_tmp)

        img_tmp[img_tmp < 0] = 0
        img_tmp[img_tmp > 1] = 0

        mask_zero = img_tmp.sum(axis=2) == 0

        img_tmp[mask_zero] = 0

        ndvi_tmp = get_NDVI_map(img_tmp)
        nir = img_tmp[:, :, 3]
        red = img_tmp[:, :, 2]

        CIred = nir / red - 1

        img_tmp[ndvi_tmp < 0.2] = 0

        img_all[:, :, :, tmp_i] = img_tmp

        Ndvi_all[:, :, tmp_i] = get_NDVI_map(img_tmp)
        NIRv_all[:, :, tmp_i] = ndvi_tmp * nir
        CIred_all[:, :, tmp_i] = CIred

        seg = felzenszwalb(img_tmp[:, :, [2, 1, 0]], scale=100, sigma=0.5, min_size=40)
        value_list, count_list = np.unique(seg, return_counts=True)
        mask_zero = morphology.remove_small_holes(mask_zero, 512, connectivity=1, in_place=False)
        mask_zero = mask_zero.astype(int)
        while True:
            if np.sum(mask_zero[seg == value_list[count_list.argmax()]]) > 1000:
                count_list[count_list.argmax()] = 0
            else:
                break

        seg = np.where(seg == value_list[count_list.argmax()], 0, 1)

        seg_cloud_mask = seg.astype(np.uint8)
        Seg_cloud_mask_all[:, :, tmp_i] = seg_cloud_mask

    img_all = img_all.astype(np.float)

    img_new = img_all

    img_new = img_new - np.nanmean(img_new, axis=(0, 1),
                                   keepdims=True)

    prc_threshold_10 = np.nanpercentile(img_new, threshold[0], axis=(0, 1, 3), keepdims=True)
    prc_threshold_90 = np.nanpercentile(img_new, threshold[1], axis=(0, 1, 3), keepdims=True)

    NIRv_mask_all = NIRv_all < 0.18

    planet_outlier_v1 = np.sum((img_new < prc_threshold_10), axis=2) > 0

    Ndvi_mask_all_v2 = Ndvi_all < 0.75
    CIred_mask = CIred_all < 6
    cloud_mask_all = np.sum((img_new > prc_threshold_90), axis=2) > 0
    planet_outlier_v2 = np.logical_and(cloud_mask_all, Seg_cloud_mask_all)

    planet_outlier = np.logical_or(planet_outlier_v2, planet_outlier_v1)

    for tmp_i in range(len(img_list)):

        img_name = img_list[tmp_i]
        img_path = os.path.join(data_dir, img_name)

        Geo_, Prj_, img_tmp = readTif_gdal(img_path)
        img_tmp = np.nan_to_num(img_tmp)

        img_tmp[img_tmp < 0] = 0
        img_tmp[img_tmp > 1] = 0

        h, w, c = img_tmp.shape

        if planet_outlier[:, :, tmp_i].shape != (h, w):
            img_tmp = cv2.resize(img_tmp, (planet_outlier[:, :, tmp_i].shape[1], planet_outlier[:, :, tmp_i].shape[0]))

        if data_type == 'S2':
            img_fourbands = img_tmp[:, :, [0, 1, 2, 6]]
        else:
            img_fourbands = img_tmp[:, :, [0, 1, 2, 3]]

        ndvi_tmp = get_NDVI_map(img_fourbands)

        green = img_fourbands[:, :, 1]
        nir = img_fourbands[:, :, 3]
        ndwi = np.divide((green - nir), (green + nir), where=(green + nir) != 0)

        img_tmp[ndvi_tmp < 0.1] = 0

        img_tmp[ndwi >= -0.35,:] = 0

        mask_raw = img_tmp.sum(axis=2) == 0

        mask_raw_v2 = mask_raw.astype(bool)

        planet_mis = planet_outlier[:, :, tmp_i]

        mis_pixmph = planet_mis.astype(bool)

        if np.sum(mis_pixmph) / mis_pixmph.reshape(-1).shape[0] > 0.9:
            continue

        mis_pixmph = morphology.remove_small_objects(mis_pixmph, min_size=32, connectivity=1,
                                                     in_place=False)

        win = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
        mis_pixmph = cv2.morphologyEx(mis_pixmph.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        img_dilate = cv2.dilate(mis_pixmph.astype(np.uint8), kernel)

        mis_pixmph = img_dilate.astype(bool)

        img_tmp[mis_pixmph, :] = 0

        mask = img_tmp.sum(axis=2) != 0
        mask_remove = morphology.remove_small_objects(mask, min_size=256, connectivity=1, in_place=False)
        img_tmp[~mask_remove, :] = 0

        save_tif(img_tmp, os.path.join(save_dir, img_name.replace('.tif', '_qc.tif')), Geo_, Prj_, img_tmp.shape[2])

        del img_tmp, mask, mask_remove, planet_mis, mis_pixmph, img_dilate

def cloud_removal_qc_gee(img_list, threshold=None, data_type='S2'):
    if threshold is None:
        threshold = [5, 95]

    img_tmp = img_list[0]
    h, w, nbands = img_tmp.shape
    nbands = 4
    img_all = np.zeros([h, w, nbands, len(img_list)])
    Ndvi_all = np.zeros([h, w, len(img_list)])
    img_all_qc = np.zeros_like(img_all)

    for tmp_i in range(len(img_list)):
        img_tmp = img_list[tmp_i]

        if data_type == 'S2':
            img_all[:, :, :, tmp_i] = img_tmp

            Ndvi_all[:, :, tmp_i] = get_NDVI_map(img_tmp)
        elif data_type == 'Landsat':
            img_all[:, :, :, tmp_i] = img_tmp[:, :, [0, 1, 2, 3]]

            Ndvi_all[:, :, tmp_i] = get_NDVI_map(img_tmp[:, :, [0, 1, 2, 3]])

    img_all = img_all.astype(np.float)

    img_new = img_all
    img_new = img_new - np.nanmean(img_new, axis=(0, 1), keepdims=True) + np.nanmean(img_new, axis=(0, 1, 3),
                                                                                     keepdims=True)

    prc_threshold_10 = np.nanpercentile(img_new, threshold[0], axis=(0, 1, 3), keepdims=True)
    prc_threshold_90 = np.nanpercentile(img_new, threshold[1], axis=(0, 1, 3), keepdims=True)

    planet_outlier_v1 = np.sum((img_new < prc_threshold_10), axis=2) > 1

    Ndvi_mask_all = Ndvi_all < 0.8
    cloud_mask_all = (np.sum((img_new > prc_threshold_90), axis=2) > 1)
    planet_outlier_v2 = np.logical_and(cloud_mask_all, Ndvi_mask_all)
    planet_outlier = np.logical_or(planet_outlier_v2, planet_outlier_v1)

    for tmp_i in range(len(img_list)):

        planet_mis = planet_outlier[:, :, tmp_i]

        mis_pixmph = planet_mis.astype(bool)
        mis_pixmph = morphology.remove_small_objects(mis_pixmph, min_size=64, connectivity=1,
                                                     in_place=False)
        win = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
        img_dilate = cv2.dilate(mis_pixmph.astype(np.uint8), kernel)

        mis_pixmph = img_dilate.astype(bool)

        img_tmp = img_list[tmp_i].astype(np.float16)

        img_tmp[mis_pixmph, :] = np.nan

        mask = np.isnan(img_tmp).sum(axis=2) == 0

        mask_remove = morphology.remove_small_objects(mask, min_size=64, connectivity=1, in_place=False)
        img_tmp[~mask_remove, :] = np.nan

        img_all_qc[:, :, :, tmp_i] = img_tmp

        del img_tmp, mask, mask_remove, planet_mis, mis_pixmph, img_dilate

    return img_all_qc

gasuss_noise = gaussian_noise
