"""Four-band ECAE site-level unmixing module.

This module implements the reference TensorFlow 1.x-compatible ECAE workflow
with the repository's local SUNSAL implementation. The training graph uses
batch statistics for normalization and applies spectral-range constraints to
the three learned endmembers.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

import unmixing

tf.disable_v2_behavior()
epsilon = 1e-8

def get_NDVI_map(img):
    nir = img[:, 3]
    red = img[:, 2]
    ndvi_map = np.divide(nir - red, nir + red, out=np.zeros_like(nir - red, dtype=np.float64), where=nir + red != 0)
    return ndvi_map

def get_EVI_map(img):
    nir = img[:, 3]
    red = img[:, 2]
    blue = img[:, 0]
    scale = 1
    evi_map = np.divide(2.5 * (nir - red), nir + 6 * red - 7.5 * blue + scale, out=np.zeros_like(nir - red, dtype=np.float64), where=nir + 6 * red - 7.5 * blue + scale != 0)
    return evi_map

def get_Grvi_map(img):
    red = img[:, 2]
    green = img[:, 1]
    griv_map = np.divide(abs(green - red), red + green + np.finfo(float).eps, out=np.zeros_like(green, dtype=np.float64), where=red + green + np.finfo(float).eps != 0)
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
    glvi_map = np.divide(2 * green - red - blue, 2 * green + red + blue, out=np.zeros_like(2 * green - red - blue, dtype=np.float64), where=2 * green + red + blue != 0)
    return glvi_map

def get_endmember_band_range_percentage(img_pln, percentile=90, percent=5):
    """
    :param img_pln:  number of pixels * nbands
    :param out_class: 'npv', 'gv', 'shade' -- the class of the endmember
    :param percentile: the percentile of the spectral reflectance of the candidates endmembers -- remove the outliers
    :param percent: the percentage of the range of the spectral reflectance of the candidates endmembers
    :return: the range of the spectral reflectance of the candidates endmembers
    """
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
        npv_ind = np.where((grvi_map < np.nanpercentile(grvi_map, npv_start)) & (evi_map < np.nanpercentile(evi_map, npv_start)) & (glvi_map < np.nanpercentile(glvi_map, npv_start)) & (ndvi_map < np.nanpercentile(ndvi_map, npv_start)))
        npv_mean = np.nanmean(img_pln[npv_ind], axis=0)
        if np.isnan(npv_mean[0]):
            npv_ind = [[0]]
            npv_start = percent
            while len(npv_ind[0]) < 10000 and npv_start < 50:
                npv_ind = np.where((grvi_map < np.nanpercentile(grvi_map, npv_start)) & (evi_map < np.nanpercentile(evi_map, npv_start)) & (glvi_map < np.nanpercentile(glvi_map, npv_start)) & (ndvi_map < np.nanpercentile(ndvi_map, npv_start)))
                npv_start += 0.5
        gv_start = percent
        gv_ind = np.where((evi_map > np.nanpercentile(evi_map, 100 - gv_start)) & (glvi_map > np.nanpercentile(glvi_map, 100 - gv_start)) & (grvi_map > np.nanpercentile(grvi_map, 100 - gv_start)) & (ndvi_map > np.nanpercentile(ndvi_map, 100 - gv_start)))
        gv_mean = np.nanmean(img_pln[gv_ind], axis=0)
        if np.isnan(gv_mean[0]):
            gv_ind = [[0]]
            gv_start = percent
            while len(gv_ind[0]) < 10000 and gv_start < 50:
                gv_ind = np.where((evi_map > np.nanpercentile(evi_map, 100 - gv_start)) & (glvi_map > np.nanpercentile(glvi_map, 100 - gv_start)) & (grvi_map > np.nanpercentile(grvi_map, 100 - gv_start)) & (ndvi_map > np.nanpercentile(ndvi_map, 100 - gv_start)))
                gv_start += 0.5
        shd_start = percent * 0.25
        Nir = img_pln[:, 3]
        shd_ind = np.where((sivi_map > np.nanpercentile(sivi_map, 100 - shd_start)) & (Nir < np.nanpercentile(Nir, shd_start)))
        shd_mean = np.nanmean(img_pln[shd_ind], axis=0)
        if np.isnan(shd_mean[0]):
            shd_ind = [[0]]
            while len(shd_ind[0]) < 5000 and shd_start < 50:
                shd_ind = np.where((sivi_map > np.nanpercentile(sivi_map, 100 - shd_start)) & (Nir < np.nanpercentile(Nir, shd_start)))
                shd_start += 0.5
    else:
        npv_ind = [[0]]
        npv_start = 0.5
        while len(npv_ind[0]) < 1000 and npv_start < 50:
            npv_ind = np.where((grvi_map < np.nanpercentile(grvi_map, npv_start)) & (evi_map < np.nanpercentile(evi_map, npv_start)) & (glvi_map < np.nanpercentile(glvi_map, npv_start)) & (ndvi_map < np.nanpercentile(ndvi_map, npv_start)))
            npv_start += 0.5
        gv_ind = [[0]]
        gv_start = 0.5
        while len(gv_ind[0]) < 1000 and gv_start < 50:
            gv_ind = np.where((evi_map > np.nanpercentile(evi_map, 100 - gv_start)) & (ndvi_map > np.nanpercentile(ndvi_map, 100 - gv_start)) & (glvi_map > np.nanpercentile(glvi_map, 100 - gv_start)) & (grvi_map > np.nanpercentile(grvi_map, 100 - gv_start)))
            gv_start += 0.5
        shd_ind = [[0]]
        shd_start = 0.5
        Nir = img_pln[:, 3]
        while len(shd_ind[0]) < 1000 and shd_start < 50:
            shd_ind = np.where((sivi_map > np.nanpercentile(sivi_map, 100 - shd_start)) & (Nir < np.nanpercentile(Nir, shd_start)))
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
    return (shade_list, npv_list, gv_list)

def init_weights(shape):
    weight = tf.Variable(tf.random_normal(shape, stddev=0.01))
    return weight

def en_net(X, w_in, beta, gama, alpha):
    l1 = tf.matmul(X, w_in)
    mean, var = tf.nn.moments(l1, axes=0, keep_dims=True)
    l1 = tf.nn.batch_normalization(l1, mean, var, beta, gama, epsilon)
    l2 = tf.nn.relu(features=l1 - alpha)
    result = tf.nn.softmax(l2)
    return (l2, result)

def de_net(X, w_out):
    out = tf.matmul(X, w_out)
    return out


def ecological_order_ok(shade, dec, ever):
    """Check the expected shade, deciduous, and evergreen spectral order."""
    return bool(
        (shade[0] <= ever[0] < dec[0])
        and (shade[1] < ever[1] <= dec[1] * 1.10)
        and (shade[2] <= ever[2] < dec[2])
    )


def train_autoencoder(Img, Img_clean, endnum, endmember_list, epoch, learning_rate, batch_size=32, require_improvement=1, delta_increasing=0.0001, begin_with_zero=True, pre_initial=False, verbose=True):
    """
    :param Img: input data for training
    :param Img_clean: the clean data for validating
    :param endnum:  the number of the endmembers
    :param endmember_list: the spectral reflectance range of the endmembers as well as the mean value
    :param epoch:  the number of the epoch
    :param learning_rate: the learning rate for training
    :param batch_size:  the batch size for training
    :param require_improvement: the number of the epoch without improvement
    :param delta_increasing: judgment for the improvement
    :param begin_with_zero: define the range of ecological constraints
    :param pre_initial: whether to use the pre-extracted endmember as the initial value
    :param verbose: whether to print the information during the training
    :return: the extracted endmembers
    """
    gv_range = endmember_list[2]
    npv_range = endmember_list[1]
    shd_range = endmember_list[0]
    Img = Img.astype(np.float32)
    Img_clean = Img_clean.astype(np.float32)
    num_of_pixels, in_size = Img_clean.shape
    Ed_tmp_initial = np.zeros((in_size, endnum))
    shade_initial = shd_range[2, :]
    dec_initial = npv_range[2, :]
    ever_initial = gv_range[2, :]
    Ed_tmp_initial[:, 0] = shade_initial
    Ed_tmp_initial[:, 1] = dec_initial
    Ed_tmp_initial[:, 2] = ever_initial
    if verbose:
        print('Initial_end_autoencoder:')
        print('Shade: ', shade_initial[[0, 1, 2, 3]])
        print('Dec: ', dec_initial[[0, 1, 2, 3]])
        print('Ever: ', ever_initial[[0, 1, 2, 3]])
    if pre_initial:
        initial_w_out = Ed_tmp_initial.transpose()
        w_out = tf.Variable(initial_value=initial_w_out.astype(np.float32))
        w_in = init_weights([in_size, endnum])
    else:
        initial_w_out = init_weights([endnum, in_size])
        w_out = init_weights([endnum, in_size])
        w_in = init_weights([in_size, endnum])
    initial_abu = get_abundance_map(Img_clean, Ed_tmp_initial)
    initial_abu = initial_abu.reshape(-1, 3)
    beta = tf.Variable(tf.zeros([1]))
    gama = tf.Variable(tf.ones([1]))
    alpha = tf.Variable(tf.zeros([1, endnum]))
    theta_ae = [w_in, w_out, beta, gama, alpha]
    end_epo = []
    end_all = []
    X = tf.placeholder(tf.float32, [None, in_size])
    Y = tf.placeholder(tf.float32, [None, in_size])
    X_test = tf.placeholder(tf.float32, [None, in_size])
    z, en_result = en_net(X, w_in, beta, gama, alpha)
    de_result = de_net(en_result, w_out)
    _, en_result_1 = en_net(X_test, w_in, beta, gama, alpha)
    de_result_2 = de_net(en_result_1, w_out)
    ae_loss = tf.losses.mean_squared_error(Y, de_result)
    AE_lr = tf.placeholder(tf.float32)
    ae_solver = tf.train.AdadeltaOptimizer(learning_rate=AE_lr).minimize(ae_loss, var_list=theta_ae)
    clip_op_all = tf.assign(w_out, tf.clip_by_value(w_out, 0, gv_range[0].max()))
    if begin_with_zero:
        clip_op_0 = tf.assign(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[0], :], tf.clip_by_value(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[0], :], 0, gv_range[0, :]))
        clip_op_1 = tf.assign(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[1], :], tf.clip_by_value(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[1], :], 0, npv_range[0, :]))
        clip_op_2 = tf.assign(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[2], :], tf.clip_by_value(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[2], :], 0, npv_range[1, :]))
    else:
        clip_op_0 = tf.assign(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[0], :], tf.clip_by_value(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[0], :], gv_range[1, :], gv_range[0, :]))
        clip_op_1 = tf.assign(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[1], :], tf.clip_by_value(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[1], :], npv_range[1, :], npv_range[0, :]))
        clip_op_2 = tf.assign(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[2], :], tf.clip_by_value(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[2], :], shd_range[1, :], shd_range[0, :]))
    img_mse = tf.losses.mean_squared_error(X_test, de_result_2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ae_lossforplt = []
        loss_all = []
        best_loss = 1
        not_decrease_record = 0
        last_improvement = 0
        index_list = np.arange(num_of_pixels)
        np.random.seed(42)
        np.random.shuffle(index_list)
        R = Img[index_list]
        R_clean = Img_clean[index_list]
        eco_step = 0
        for step in range(epoch):
            for i in range(num_of_pixels // batch_size + 1):
                if i < num_of_pixels // batch_size:
                    batch_x = R[i * batch_size:i * batch_size + batch_size]
                    batch_y = R_clean[i * batch_size:i * batch_size + batch_size]
                else:
                    batch_x = R[i * batch_size:]
                    batch_y = R_clean[i * batch_size:]
                ae_solver_, ae_loss_ = sess.run([ae_solver, ae_loss], feed_dict={X: batch_x, AE_lr: learning_rate, Y: batch_y})
                sess.run(clip_op_all)
                sess.run(clip_op_0)
                sess.run(clip_op_1)
                sess.run(clip_op_2)
            if verbose:
                print('Step: {}'.format(step))
                print('AE_loss: {:.5}'.format(ae_loss_))
                print('Learning rate: {:.5}'.format(learning_rate))
            en_result_, w_out_, final_loss_auto = sess.run([en_result_1, w_out, img_mse], feed_dict={X_test: Img_clean})
            if verbose:
                print('Image Loss Value :', final_loss_auto)
            ae_loss_ = final_loss_auto
            Ed_tmp = w_out_.transpose()
            Ed_tmp = Ed_tmp[:, Ed_tmp[3, :].argsort()]
            shade_ = Ed_tmp[:, 0]
            dec_ = Ed_tmp[:, 1]
            ever_ = Ed_tmp[:, 2]
            if verbose:
                print('Shade: ', shade_[[0, 1, 2, 3]])
                print('Dec: ', dec_[[0, 1, 2, 3]])
                print('Ever: ', ever_[[0, 1, 2, 3]])
            loss_all.append(ae_loss_)
            end_all.append(w_out_)
            if ecological_order_ok(shade_, dec_, ever_):
                ae_lossforplt.append(final_loss_auto)
                end_epo.append(w_out_)
                eco_step += 1
                if verbose:
                    print('Match ecological ordering')
            else:
                eco_step = 0
                if verbose:
                    print('Ecological ordering rejected')
                continue
            if step > 0:
                if ae_loss_ < loss_all[step - 1]:
                    if ae_loss_ < best_loss - delta_increasing:
                        not_decrease_record = 0
                        best_loss = ae_loss_
                        last_improvement = 0
                        if verbose:
                            print('\nBest loss value: ', best_loss, '\n')
                    else:
                        last_improvement += 1
                    if eco_step > 10:
                        if last_improvement > require_improvement - 1:
                            if verbose:
                                print('Early Stop')
                            break
                else:
                    if verbose:
                        print('loss value do not decrease')
                    break
    tf.keras.backend.clear_session()
    if len(ae_lossforplt) != 0:
        min_index = ae_lossforplt.index(min(ae_lossforplt))
        final_end = end_epo[min_index].transpose()
    elif shade_initial[1] <= ever_initial[1] < dec_initial[1] * 1.1:
        final_end = Ed_tmp_initial.transpose()
    else:
        constant_w_out = tf.placeholder(tf.float32, shape=(initial_w_out.shape[0], initial_w_out.shape[1]))
        tf_img = tf.placeholder(tf.float32, shape=(Img_clean.shape[0], Img_clean.shape[1]))
        tf_abu_init = tf.placeholder(tf.float32, shape=(initial_abu.shape[0], initial_abu.shape[1]))
        de_result_initial = tf.matmul(tf_abu_init, constant_w_out)
        final_loss_initial_ = tf.losses.mean_squared_error(tf_img, de_result_initial)
        with tf.Session() as sess2:
            sess2.run(tf.global_variables_initializer())
            final_loss_initial = sess2.run(final_loss_initial_, feed_dict={tf_img: Img_clean.astype(np.float32), tf_abu_init: initial_abu, constant_w_out: Ed_tmp_initial.transpose()})
        min_index = loss_all.index(min(loss_all))
        final_end = end_all[min_index].transpose()
        if loss_all[min_index] > final_loss_initial:
            final_end = initial_w_out.transpose()
    new_rank_index = final_end[3, :].argsort()
    final_end = final_end[:, new_rank_index]
    shade_ = final_end[:, 0]
    dec_ = final_end[:, 1]
    ever_ = final_end[:, 2]
    if verbose:
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
