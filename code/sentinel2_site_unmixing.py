"""Sentinel-2 site-level unmixing workflow.

"""

# In[] Imports
import matplotlib.pyplot as plt
from ecae_modules_four_bands import get_endmember_band_range_percentage, train_autoencoder, plot_endmember
import unmixing
from data_preprocessing import readTif_gdal, save_tif, gaussian_noise
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib

# In[] Workflow
matplotlib.use("Agg")

# In[] Functions
def get_abundance_map(image, endmember_mean):

    if len(image.shape) == 3:
        height, width, bands = image.shape
        input_img = image.reshape(height * width, bands)
    else:
        input_img = image

    mask_ind = np.all(input_img != 0, axis=1)

    new_input_img = input_img[mask_ind].transpose([1, 0])

    img_abundance = unmixing.sunsal(endmember_mean, new_input_img, al_iters=200, positivity=True, addone=True, verbose=False)[0].transpose([1, 0])

    final_abu = np.zeros((input_img.shape[0], endmember_mean.shape[1]))

    final_abu[mask_ind] = img_abundance

    if len(image.shape) == 3:
        final_abu = final_abu.reshape(height, width, 3)

    return final_abu

def sentinel_2_multiyears_unmixing():
    endnum = 3
    epoch = 5
    learning_rate = 1e-3
    batch_size = 512
    delta_increasing = 1e-4
    require_improvement = 1
    begin_with_zero = False
    pre_initial = True
    verbose = True

    data_dir = r'data/sentinel2/rja_shared'
    Save_Dir = r'outputs/sentinel2_unmixing'

    site_list = ['Testing']

    if not os.path.exists(Save_Dir):
        os.makedirs(Save_Dir)

    for site in site_list:

        data_folder = os.path.join(data_dir, site)
        save_folder_dir = os.path.join(Save_Dir, site)

        year_list = [x for x in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, x)) if x[0] == '2']

        for year in year_list:

            data_path = os.path.join(data_folder, year)
            save_path = os.path.join(save_folder_dir, year)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            data_list = [x for x in os.listdir(data_path) if year in x if x.endswith('.tif')]

            for index, img_name in enumerate(data_list):
                img_path = os.path.join(data_path, img_name)
                Geo_, Prj_, img = readTif_gdal(img_path)

                img = img[:, :, [0, 1, 2, 3]]

                img = img.astype(np.float32)
                img = np.nan_to_num(img)
                img[img < 0] = 0
                img[img > 1] = 0

                if index == 0:
                    print(img.shape)

                nir = img[:, :, 3]
                red = img[:, :, 2]
                blue = img[:, :, 0]

                evi_map = np.divide(2.5 * (nir - red), (nir + 6 * red - 7.5 * blue + 1),
                                    out=np.zeros_like((nir - red), dtype=np.float64),
                                    where=(nir + 6 * red - 7.5 * blue + 1) != 0)

                non_ind_1 = np.where(evi_map < 0.15)

                img_res = img.reshape(-1, img.shape[2])
                img_res = np.nan_to_num(img_res)

                mask_ind = np.sum(img_res, axis=1) != 0

                img_s2_input = img_res[mask_ind]

                try:
                    _, img_s2_test = train_test_split(img_s2_input, test_size=0.8, random_state=42)
                    img_s2_input = img_s2_test
                    if index == 0:
                        img_all = img_s2_input

                    else:
                        img_all = np.vstack([img_all, img_s2_input])

                except ValueError:
                    print('Error')

            tmp_threshold = 95
            percent = 10
            shd_range, npv_range, gv_range = get_endmember_band_range_percentage(img_all,
                                                                                 percentile=tmp_threshold,
                                                                                 percent=percent)

            Ed_tmp_initial = np.zeros((4, endnum))
            shade_initial = shd_range[2, :]
            dec_initial = npv_range[2, :]
            ever_initial = gv_range[2, :]

            Ed_tmp_initial[:, 0] = shade_initial
            Ed_tmp_initial[:, 1] = dec_initial
            Ed_tmp_initial[:, 2] = ever_initial

            plot_endmember(Ed_tmp_initial, site + '_' + str(year) + '_Index_Endmember')

            plt.savefig(os.path.join(save_path, site + '_' + str(year) + '_Index_Endmember.png'), dpi=300)
            plt.close()

            print('Max GV:{}\nMax NPV:{}\nMax Shade:{}'.format(gv_range[:2, :], npv_range[:2, :], shd_range[:2, :]))
            endmember_range = [shd_range, npv_range, gv_range]

            print(img_all.shape)

            input_end = train_autoencoder(
                Img=img_all,
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
                verbose=True, )

            input_end = input_end[:,input_end[3, :].argsort()]

            plot_endmember(input_end, site + '_' + str(year) + '_' + 'Endmember')

            plt.savefig(os.path.join(save_path, site + '_' + str(year) + '_Endmember.png'), dpi=300)
            plt.close()

            for img_name in data_list:
                data_img_path = os.path.join(data_path, img_name)
                Geo_, Prj_, img = readTif_gdal(data_img_path)

                img = img[:, :, [0, 1, 2, 3]]

                img = img.astype(np.float32)
                img = np.nan_to_num(img)
                img[img < 0] = 0
                img[img > 1] = 0

                nir = img[:, :, 3]
                red = img[:, :, 2]

                img_res = img.reshape(-1, img.shape[2])

                img_res = np.nan_to_num(img_res)

                mask_ind = np.sum(img_res, axis=1) != 0
                img_input = img_res[mask_ind]

                abu_map = get_abundance_map(img_input, input_end)
                final_abu_map = np.zeros((img_res.shape[0], 3))
                final_abu_map[mask_ind] = abu_map
                out_put_abu = final_abu_map.reshape(img.shape[0], img.shape[1], endnum)

                abu_name = img_name.replace('.tif', '_abu.tif')
                save_img_path = os.path.join(save_path, abu_name)
                save_tif(out_put_abu, save_img_path, Geo_, Prj_, endnum)

# In[] Main
if __name__ == '__main__':

    sentinel_2_multiyears_unmixing()
