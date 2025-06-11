
# This code is part of a project that processes satellite imagery data to derive deciduousness maps using an autoencoder model.
# Author: Guangqin Song

# -*- coding: utf-8 -*-
import io
# import multiprocessing
import time
import warnings
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from S1_Data_Preprocessing_amazon import  cloud_removal_qc_gee_superpixel, cloud_removal_qc_gee, readTif_gdal, save_tif_int
from S2_ECAE_Modules_Four_Bands_v2 import get_endmember_band_range_percentage, train_autoencoder, plot_endmember, get_abundance_map
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=Warning)
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    import gdal
except:
    from osgeo import gdal

gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.UseExceptions()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁止gpu


# In[]
year = 2020
Dara_Dir = r'W:\Song_Amazon_Mapping\Amazon_S2_Data_{}'.format(year)
Save_Dir = r'W:\Song_Amazon_Mapping\Results\Amazon_{}'.format(year)

Grid_Num = 76276

def autoencoder_multiprocess(grid_id):
    # 1. read the time-series data
    # 2. cloud removal
    # 3. get the candidate endmembers
    # 4. train the autoencoder model
    # 5. unmixing the data and save the results

    data_folder = os.path.join(Dara_Dir, 'Amazon_Grid_{}'.format(grid_id))
    save_folder = os.path.join(Save_Dir, 'Amazon_Grid_{}'.format(grid_id))


    endnum = 3
    epoch = 30
    learning_rate = 1e-3
    batch_size = 512
    delta_increasing = 1e-4
    require_improvement = 1
    begin_with_zero = False
    pre_initial = True
    verbose = False

    data_list = [x for x in os.listdir(data_folder) if x.endswith('.tif')]

    img_list = []
    data_new_name = []
    for data_name in data_list:
        data_path = os.path.join(data_folder, data_name)
        # _, _, img_tmp = readTif_gdal(data_path)
        try:
            _, _, img_tmp = readTif_gdal(data_path)
        except Exception as e:
            print('Error image: ', data_name)
            print('Error: ', e)
            continue

        img_tmp = img_tmp / 10000

        mask_nan = np.sum(img_tmp, axis=2) == 0
        if np.sum(mask_nan.astype(int)) / (img_tmp.shape[0] * img_tmp.shape[1]) < 0.95:
            img_list.append(img_tmp[:, :, [0, 1, 2, 6]])
            data_new_name.append(data_name)

    if len(img_list) > 0:
        star = time.time()
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for tmp_i in range(len(img_list)):
            # img_tmp = img_np_all[:, :, :, tmp_i]
            img_tmp = img_list[tmp_i]
            img_tmp = np.nan_to_num(img_tmp)

            img_res = img_tmp.reshape(-1, img_tmp.shape[2])

            mask_ind = np.sum(img_res, axis=1) != 0  # ~(img_res == 0).all(1)  # remove the background
            img_s2_input = img_res[mask_ind]

            try:
                _, img_s2_test = train_test_split(img_s2_input, test_size=0.8, random_state=42)
                img_s2_input = img_s2_test
                if tmp_i == 0:
                    img_all = img_s2_input
                else:
                    img_all = np.vstack([img_all, img_s2_input])
            except ValueError:
                print('Error')

        img_all = img_all.astype(np.float32)
        # obtain the initial endmembers
        tmp_threshold = 95  # remove the outline pixels
        percent = 10  # the percentage to extract image
        shd_range, npv_range, gv_range = get_endmember_band_range_percentage(img_all, percentile=tmp_threshold, percent=percent)
        # shade_list, npv_list, gv_list
        Ed_tmp_initial = np.zeros((4, 3))

        Ed_tmp_initial[:, 0] = shd_range[2, :]
        Ed_tmp_initial[:, 1] = npv_range[2, :]
        Ed_tmp_initial[:, 2] = gv_range[2, :]

        # plot_endmember(Ed_tmp_initial, 'Amazon_Grid_' + str(grid_id) + '_Index_Endmember')
        # plt.savefig(os.path.join(save_folder, 'Amazon_Grid_' + str(grid_id) + '_Index_Endmember.png'), dpi=300)
        # plt.close()
        endmember_range = [shd_range, npv_range, gv_range]

        print('Begin to train model at grid {}'.format(grid_id))
        # train the autoencoder model'
        input_end = train_autoencoder(
            Img=img_all,  # img_all_noise, #img_all
            Img_clean=img_all,
            endnum=endnum,
            endmember_list=endmember_range,
            epoch=epoch,
            learning_rate=learning_rate,
            batch_size=batch_size,
            require_improvement=require_improvement,
            delta_increasing=delta_increasing,
            begin_with_zero=begin_with_zero,
            pre_initial=pre_initial,
            verbose=verbose,
        )

        # plot_endmember(input_end, 'Amazon_Grid_' + str(grid_id) + '_Endmember')
        # plt.savefig(os.path.join(save_folder, 'Amazon_Grid_' + str(grid_id) + '_Endmember.png'), dpi=300)
        # plt.close()

        for tmp_i, data_name in enumerate(data_new_name): #data_new_name
            data_path = os.path.join(data_folder, data_name)
            Geo_, Prj_, img_tmp = readTif_gdal(data_path)

            # img = img_np_all[:, :, :, tmp_i]
            img = img_tmp[:, :, [0, 1, 2, 6]]/10000.0
            img = np.nan_to_num(img)

            img[img < 0] = 0
            img[img > 1] = 0

            img_res = img.reshape(-1, img.shape[2])

            mask_ind = np.sum(img_res, axis=1) != 0  # ~(img_res == 0).all(1)

            img_input = img_res[mask_ind]

            abu_map = get_abundance_map(img_input, input_end)
            final_abu_map = np.ones((img_res.shape[0], 3)) * 9999
            final_abu_map[mask_ind] = np.array(abu_map * 1000).astype(int)
            out_put_abu = final_abu_map.reshape(img.shape[0], img.shape[1], endnum)

            abu_name = data_name.replace('.tif', '_abu.tif')
            save_img_path = os.path.join(save_folder, abu_name)
            save_tif_int(out_put_abu, save_img_path, Geo_, Prj_, endnum)
        print('Finish Grid: ', grid_id, ' Time: ', time.time() - star)
    else:
        os.makedirs(save_folder + '_empty')
        return 0

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]

def deciduousness_mapping():
    # Grid_Num = 20
    with ProcessPoolExecutor(max_workers=10) as outer_pool:
        group_list_all = np.arange(Grid_Num)
        for tile_id in group_list_all:# group_list_all:# reverse_group:  # tile_num #group_list_all:#
            outer_pool.submit(autoencoder_multiprocess, int(tile_id))


if __name__ == '__main__':
    deciduousness_mapping()
