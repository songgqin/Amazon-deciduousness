"""IG-ECAE four-band unmixing modules.

"""

# In[] Imports
import sys
import warnings
import numpy as np
import tensorflow as tf
import unmixing
import matplotlib.pyplot as plt
import gc

# In[] Workflow
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append(r'/')

epsilon = 1e-8

# In[] Functions
def get_NDVI_map(img):
    nir = img[:, 3]
    red = img[:, 2]
    ndvi_map = np.divide((nir - red), (nir + red), out=np.zeros_like((nir - red), dtype=np.float64),
                         where=(nir + red) != 0)
    return ndvi_map

def get_EVI_map(img):

    nir = img[:, 3]
    red = img[:, 2]
    blue = img[:, 0]
    scale = 1
    evi_map = np.divide(2.5 * (nir - red), (nir + 6 * red - 7.5 * blue + scale),
                        out=np.zeros_like((nir - red), dtype=np.float64),
                        where=(nir + 6 * red - 7.5 * blue + scale) != 0)
    return evi_map

def get_CIred_map(img):
    nir = img[:, 3]
    red = img[:, 2]
    CIred = np.divide(nir, red, out=np.zeros_like(red, dtype=np.float64), where=(red != 0)) - 1
    return CIred

def get_GCC_map(img):
    red = img[:, 2]
    blue = img[:, 0]
    green = img[:, 1]
    gcc = np.divide(green, (red + blue + green), out=np.zeros_like(red + blue + green, dtype=np.float64),
                    where=((red + blue + green) != 0))

    return gcc

def get_Grvi_map(img):
    red = img[:, 2]
    green = img[:, 1]
    '''
    griv_map = np.divide(green, red, out=np.zeros_like(green, dtype=np.float64),
                         where=red != 0)
    '''
    griv_map = np.divide(abs(green - red), (red + green + np.finfo(float).eps),
                         out=np.zeros_like(green, dtype=np.float64),
                         where=(red + green + np.finfo(float).eps) != 0)

    return griv_map

def get_Sivi_map(img):
    blue = img[:, 0]
    green = img[:, 1]
    red = img[:, 2]

    sivi_map = np.power((1 - blue) * (1 - green) * (1 - red), 1 / 3)

    return sivi_map

def get_Glvi_map(img):
    blue = img[:, 0]
    green = img[:, 1]
    red = img[:, 2]
    nir = img[:, 3]

    glvi_map = np.divide((2 * green - red - blue), (2 * green + red + blue),
                         out=np.zeros_like((2 * green - red - blue), dtype=np.float64),
                         where=(2 * green + red + blue) != 0)

    return glvi_map

def get_NIRv_map(img):
    red = img[:, 2]
    nir = img[:, 3]
    NIRv_map = nir * (nir - red) / (nir + red)

    return NIRv_map

def get_endmember_band_range_percentage(img_pln, percentile=90, percent=5):
    evi_map = get_EVI_map(img_pln)
    error_ind = np.where((evi_map > 1) | (evi_map < 0))
    img_pln = np.delete(img_pln, error_ind[0], axis=0)
    evi_map = np.delete(evi_map, error_ind[0], axis=0)
    grvi_map = get_Grvi_map(img_pln)
    ndvi_map = get_NDVI_map(img_pln)
    sivi_map = get_Sivi_map(img_pln)
    glvi_map = get_Glvi_map(img_pln)

    img_res = img_pln
    mask = (img_res == 0).all(1)
    evi_map = evi_map.reshape(-1)[~mask]
    grvi_map = grvi_map.reshape(-1)[~mask]
    ndvi_map = ndvi_map.reshape(-1)[~mask]
    sivi_map = sivi_map.reshape(-1)[~mask]
    glvi_map = glvi_map.reshape(-1)[~mask]
    img_pln = img_res[~mask]

    if percent:

        npv_start = percent
        npv_ind = np.where(
            (grvi_map < np.nanpercentile(grvi_map, npv_start)) &
            (evi_map < np.nanpercentile(evi_map, npv_start)) &
            (glvi_map < np.nanpercentile(glvi_map, npv_start)) &
            (ndvi_map < np.nanpercentile(ndvi_map, npv_start))
        )
        npv_mean = np.nanmean(img_pln[npv_ind], axis=0)

        if np.isnan(npv_mean[0]):
            npv_ind = [[0]]
            npv_start = percent
            while len(npv_ind[0]) < 10000 and npv_start < 50:
                npv_ind = np.where(
                    (grvi_map < np.nanpercentile(grvi_map, npv_start)) &
                    (evi_map < np.nanpercentile(evi_map, npv_start)) &
                    (glvi_map < np.nanpercentile(glvi_map, npv_start)) &
                    (ndvi_map < np.nanpercentile(ndvi_map, npv_start))
                )
                npv_start += 0.5

        gv_start = percent
        gv_ind = np.where(
            (evi_map > np.nanpercentile(evi_map, 100 - gv_start)) &
            (glvi_map > np.nanpercentile(glvi_map, 100 - gv_start)) &
            (grvi_map > np.nanpercentile(grvi_map, 100 - gv_start)) &
            (ndvi_map > np.nanpercentile(ndvi_map, 100 - gv_start))

        )
        gv_mean = np.nanmean(img_pln[gv_ind], axis=0)

        if np.isnan(gv_mean[0]):
            gv_ind = [[0]]
            gv_start = percent
            while len(gv_ind[0]) < 10000 and gv_start < 50:
                gv_ind = np.where(
                    (evi_map > np.nanpercentile(evi_map, 100 - gv_start)) &
                    (glvi_map > np.nanpercentile(glvi_map, 100 - gv_start)) &
                    (grvi_map > np.nanpercentile(grvi_map, 100 - gv_start)) &
                    (ndvi_map > np.nanpercentile(ndvi_map, 100 - gv_start))
                )
                gv_start += 0.5

        shd_start = percent * 0.25
        Nir = img_pln[:, 3]
        shd_ind = np.where(
            (sivi_map > np.nanpercentile(sivi_map, 100 - shd_start)) &
            (Nir < np.nanpercentile(Nir, shd_start)))
        shd_mean = np.nanmean(img_pln[shd_ind], axis=0)

        if np.isnan(shd_mean[0]):
            shd_ind = [[0]]
            while len(shd_ind[0]) < 5000 and shd_start < 50:
                shd_ind = np.where(
                    (sivi_map > np.nanpercentile(sivi_map, 100 - shd_start)) &
                    (Nir < np.nanpercentile(Nir, shd_start)))
                shd_start += 0.5

    else:
        npv_ind = [[0]]
        npv_start = 0.5
        while len(npv_ind[0]) < 1000 and npv_start < 50:
            npv_ind = np.where(
                (grvi_map < np.nanpercentile(grvi_map, npv_start)) &
                (evi_map < np.nanpercentile(evi_map, npv_start)) &
                (glvi_map < np.nanpercentile(glvi_map, npv_start)) &
                (ndvi_map < np.nanpercentile(ndvi_map, npv_start))
            )
            npv_start += 0.5

        gv_ind = [[0]]
        gv_start = 0.5
        while len(gv_ind[0]) < 1000 and gv_start < 50:
            gv_ind = np.where(
                (evi_map > np.nanpercentile(evi_map, 100 - gv_start)) &
                (ndvi_map > np.nanpercentile(ndvi_map, 100 - gv_start)) &
                (glvi_map > np.nanpercentile(glvi_map, 100 - gv_start)) &
                (grvi_map > np.nanpercentile(grvi_map, 100 - gv_start))
            )
            gv_start += 0.5

        shd_ind = [[0]]
        shd_start = 0.5
        Nir = img_pln[:, 3]
        while len(shd_ind[0]) < 1000 and shd_start < 50:
            shd_ind = np.where(
                (sivi_map > np.nanpercentile(sivi_map, 100 - shd_start)) &
                (Nir < np.nanpercentile(Nir, shd_start)))
            shd_start += 0.5

    npv_mean = np.nanmean(img_pln[npv_ind], axis=0)

    npv_maximum = np.nanpercentile(img_pln[npv_ind], percentile, axis=0)
    npv_minimum = np.nanpercentile(img_pln[npv_ind], 100 - percentile, axis=0)
    npv_list = np.array((npv_maximum, npv_minimum, npv_mean))

    gv_mean = np.nanmean(img_pln[gv_ind], axis=0)

    gv_maximum = np.nanpercentile(img_pln[gv_ind], percentile, axis=0)
    gv_minimum = np.nanpercentile(img_pln[gv_ind], 100 - percentile, axis=0)
    gv_list = np.array((gv_maximum, gv_minimum, gv_mean))

    shd_mean = np.nanmean(img_pln[shd_ind], axis=0)
    shd_maximum = np.nanpercentile(img_pln[shd_ind], percentile, axis=0)
    shd_minimum = np.nanpercentile(img_pln[shd_ind], 100 - percentile, axis=0)

    shade_list = np.array((shd_maximum, shd_minimum, shd_mean))

    return shade_list, npv_list, gv_list

def init_weights(shape):

    weight = np.random.normal(0,0.01, shape)
    return weight

class EndmemberWeightConstraint(tf.keras.constraints.Constraint):

    def __init__(self, num_endmembers, num_bands, min_values, max_values):
        self.num_endmembers = num_endmembers
        self.num_bands = num_bands
        self.min_values = min_values
        self.max_values = max_values

    def __call__(self, w):
        constrained_weights = []

        for i in range(self.num_endmembers):
            band_weight = w[tf.nn.top_k(w[:, 3], k=3).indices[i], :]
            constrained_weights.append(tf.clip_by_value(band_weight, self.min_values[i, :], self.max_values[i, :]))

        return tf.stack(constrained_weights)

def IG_ECAE_Model(input_dim, endnum, weight_constraint, initial_weights):

    input_img = tf.keras.Input(shape=(input_dim,))

    encoder_l1 = tf.keras.layers.Dense(endnum, use_bias=True)(input_img)
    encoder_bn = tf.keras.layers.BatchNormalization()(encoder_l1)
    encoder_dens = tf.keras.layers.Dense(endnum, use_bias=True, activation='relu',
                                         kernel_constraint=tf.keras.constraints.NonNeg())(encoder_bn)

    result = tf.keras.layers.Softmax()(encoder_dens)

    decoded = tf.keras.layers.Dense(input_dim,
                                    activation='linear',
                                    kernel_initializer=tf.keras.initializers.Constant(value=initial_weights),
                                    use_bias='False',
                                    kernel_constraint=weight_constraint,
                                    name='endmember')(result)

    autoencoder = tf.keras.Model(input_img, decoded)

    return autoencoder

def train_autoencoder(Img, Img_clean,
                      endnum,
                      endmember_list,
                      epoch,
                      learning_rate,
                      batch_size=32,
                      require_improvement=1,
                      delta_increasing=1e-4,
                      begin_with_zero=True,
                      pre_initial=False,
                      verbose=True,
                      ):

    Img = Img.astype(np.float32)
    Img_clean = Img_clean.astype(np.float32)
    num_of_pixels, in_size = Img_clean.shape

    gv_range = endmember_list[2]
    npv_range = endmember_list[1]
    shd_range = endmember_list[0]

    min_values = np.zeros((endnum, in_size))
    max_values = np.zeros((endnum, in_size))

    if begin_with_zero:
        min_values[0, :] = 0
        min_values[1, :] = 0
        min_values[2, :] = 0
    else:
        min_values[0, :] = gv_range[1, :]
        min_values[1, :] = npv_range[1, :]
        min_values[2, :] = shd_range[1, :]

    max_values[0, :] = gv_range[0, :]
    max_values[1, :] = npv_range[0, :]
    max_values[2, :] = shd_range[0, :]

    weight_constraint = EndmemberWeightConstraint(endnum, in_size, min_values, max_values)

    Ed_tmp_initial = np.zeros((in_size, endnum))
    shade_initial = shd_range[2, :]
    dec_initial = npv_range[2, :]
    ever_initial = gv_range[2, :]

    Ed_tmp_initial[:, 0] = shade_initial
    Ed_tmp_initial[:, 1] = dec_initial
    Ed_tmp_initial[:, 2] = ever_initial

    if pre_initial:

        w_out = Ed_tmp_initial.transpose()

    else:

        w_out = init_weights([endnum, in_size])

    IG_ECAE = IG_ECAE_Model(in_size, endnum, weight_constraint, w_out)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=require_improvement,
                                                min_delta=delta_increasing, mode='min')

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    IG_ECAE.compile(optimizer=optimizer, loss='mse')

    IG_ECAE.fit(Img, Img_clean, epochs=epoch, batch_size=batch_size, verbose=verbose, shuffle=True,
                callbacks=[callback])

    endmember = IG_ECAE.get_layer('endmember').get_weights()[0]

    gc.collect()
    tf.keras.backend.clear_session()

    final_end = endmember.transpose()

    new_rank_index = final_end[3, :].argsort()
    final_end = final_end[:, new_rank_index]

    ever_ = final_end[:, 2]
    dec_ = final_end[:, 1]
    shade_ = final_end[:, 0]

    print('final_end_autoencoder:')
    print('Shade: ', shade_[[0, 1, 2, 3]])
    print('Dec: ', dec_[[0, 1, 2, 3]])
    print('Ever: ', ever_[[0, 1, 2, 3]])

    print('Initial_end_autoencoder:')
    print('Shade: ', shade_initial[[0, 1, 2, 3]])
    print('Dec: ', dec_initial[[0, 1, 2, 3]])
    print('Ever: ', ever_initial[[0, 1, 2, 3]])

    return final_end

def get_abundance_map(image, endmember_mean):

    input_img = image

    img_abundance = unmixing.sunsal(endmember_mean, input_img, al_iters=100,
                                       positivity=True, addone=True, verbose=False)[0].transpose([1, 0])

    return img_abundance

def plot_endmember(Ed_mean, title):
    fig = plt.figure(figsize=(6, 4))

    x = np.arange(Ed_mean.shape[0])

    plt.plot(x, Ed_mean[:, 2], color='g', label='Evergreen')
    plt.plot(x, Ed_mean[:, 1], color='r', label='Deciduous')
    plt.plot(x, Ed_mean[:, 0], color='k', label='Shade')

    plt.ylim(0, 0.6)
    plt.xticks(x)
    plt.title(title)
    plt.legend()

    return fig
