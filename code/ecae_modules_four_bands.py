"""Four-band ECAE site-level unmixing module.

This module keeps the reference TensorFlow 1.x-compatible ECAE training
workflow while using the repository's local SUNSAL implementation.
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
    ndvi_map = np.divide((nir - red), (nir + red), out=np.zeros_like((nir - red), dtype=np.float64),
                         where=(nir + red) != 0)
    return ndvi_map

def get_EVI_map(img):
    # img[np.where(img == 0)] = np.nan
    nir = img[:, 3]
    red = img[:, 2]
    blue = img[:, 0]
    scale = 1
    evi_map = np.divide(2.5 * (nir - red), (nir + 6 * red - 7.5 * blue + scale),
                        out=np.zeros_like((nir - red), dtype=np.float64),
                        where=(nir + 6 * red - 7.5 * blue + scale) != 0)
    return evi_map

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

    # v = (blue + green + red) / 3
    # s = 1 - np.divide(np.minimum(np.minimum(blue, green), red), v, out=np.zeros_like(v, dtype=np.float64),
    #                    where=v != 0)

    # sivi_map = s-v/(s+v)

    sivi_map = np.power((1 - blue) * (1 - green) * (1 - red), 1 / 3)

    return sivi_map

def get_Glvi_map(img):
    blue = img[:, 0]
    green = img[:, 1]
    red = img[:, 2]
    nir = img[:, 3]
    # glvi_map =nir*(nir-red)/(nir+red)
    glvi_map = np.divide((2 * green - red - blue), (2 * green + red + blue),
                         out=np.zeros_like((2 * green - red - blue), dtype=np.float64),
                         where=(2 * green + red + blue) != 0)

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
    # remove background
    img_res = img_pln
    mask = (img_res == 0).all(1)
    evi_map = evi_map.reshape(-1)[~mask]
    grvi_map = grvi_map.reshape(-1)[~mask]
    ndvi_map = ndvi_map.reshape(-1)[~mask]
    sivi_map = sivi_map.reshape(-1)[~mask]
    glvi_map = glvi_map.reshape(-1)[~mask]
    img_pln = img_res[~mask]

    # candidate endmember selections based on indices and percentage
    # npv endmember
    if percent:
        # NPV
        npv_start = percent
        npv_ind = np.where(
            (grvi_map < np.nanpercentile(grvi_map, npv_start)) &
            (evi_map < np.nanpercentile(evi_map, npv_start)) &
            (glvi_map < np.nanpercentile(glvi_map, npv_start)) &
            (ndvi_map < np.nanpercentile(ndvi_map, npv_start))
        )
        npv_mean = np.nanmean(img_pln[npv_ind], axis=0)

        # if len(npv_ind[0]) > 10000:
        #     npv_ind = [[0]]
        #     npv_start = 0.5
        #     while len(npv_ind[0]) < 10000 and npv_start < 50:
        #         npv_ind = np.where(
        #             # (grvi_map < np.nanpercentile(grvi_map, npv_start)) &
        #             (evi_map < np.nanpercentile(evi_map, npv_start)) &
        #             # (glvi_map < np.nanpercentile(glvi_map, npv_start)) &
        #             (ndvi_map < np.nanpercentile(ndvi_map, npv_start))
        #         )
        #         npv_start += 0.5

        # if there are no pixels in the npv class, then increase the percentage of the range
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
        # GV
        gv_start = percent
        gv_ind = np.where(
            (evi_map > np.nanpercentile(evi_map, 100 - gv_start)) &
            (glvi_map > np.nanpercentile(glvi_map, 100 - gv_start)) &
            (grvi_map > np.nanpercentile(grvi_map, 100 - gv_start)) &
            (ndvi_map > np.nanpercentile(ndvi_map, 100 - gv_start))

        )
        gv_mean = np.nanmean(img_pln[gv_ind], axis=0)

        # if len(gv_ind[0]) > 10000:
        #     gv_ind = [[0]]
        #     gv_start = 0.5
        #     while len(gv_ind[0]) < 5000 and gv_start < 50:
        #         gv_ind = np.where(
        #             (evi_map > np.nanpercentile(evi_map, 100 - gv_start)) &
        #             # (glvi_map > np.nanpercentile(glvi_map, 100 - gv_start)) &
        #             # (grvi_map > np.nanpercentile(grvi_map, 100 - gv_start)) &
        #             (ndvi_map > np.nanpercentile(ndvi_map, 100 - gv_start))
        #         )
        #         gv_start += 0.5

        # if there are no pixels in the gv class, then increase the percentage of the range
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
        # shade
        shd_start = percent*0.25
        Nir = img_pln[:, 3]
        shd_ind = np.where(
            (sivi_map > np.nanpercentile(sivi_map, 100 - shd_start)) &
            (Nir < np.nanpercentile(Nir, shd_start)))
        shd_mean = np.nanmean(img_pln[shd_ind], axis=0)

        # if len(shd_ind[0]) > 1000:
        #     shd_ind = [[0]]
        #     shd_start = 0.5
        #     while len(shd_ind[0]) < 1000 and shd_start < 50:
        #         shd_ind = np.where(
        #             (sivi_map > np.nanpercentile(sivi_map, 100 - shd_start)) &
        #             (Nir < np.nanpercentile(Nir, shd_start)))
        #         shd_start += 0.5

        # if there are no pixels in the shd class, then increase the percentage of the range
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
    # obtain the spectral reflectance range of the npv class based on the percentile
    npv_maximum = np.nanpercentile(img_pln[npv_ind], percentile, axis=0)
    npv_minimum = np.nanpercentile(img_pln[npv_ind], 100 - percentile, axis=0)
    npv_list = np.array((npv_maximum, npv_minimum, npv_mean))

    # gv endmembers
    gv_mean = np.nanmean(img_pln[gv_ind], axis=0)
    # obtain the spectral reflectance range of the gv class based on the percentile
    gv_maximum = np.nanpercentile(img_pln[gv_ind], percentile, axis=0)
    gv_minimum = np.nanpercentile(img_pln[gv_ind], 100 - percentile, axis=0)
    gv_list = np.array((gv_maximum, gv_minimum, gv_mean))

    # shade endmembers
    shd_mean = np.nanmean(img_pln[shd_ind], axis=0)
    shd_maximum = np.nanpercentile(img_pln[shd_ind], percentile, axis=0)
    shd_minimum = np.nanpercentile(img_pln[shd_ind], 100 - percentile, axis=0)

    # extract the minimum and maximum values of shade from three classes
    # shd_minimum_v2 = np.minimum(np.minimum(shd_minimum, npv_minimum), gv_minimum)
    # shd_maximum_v2 = np.minimum(np.minimum(shd_maximum, npv_maximum), gv_maximum)

    shade_list = np.array((shd_maximum, shd_minimum, shd_mean))
    # shade_list = np.array((shd_maximum_v2, shd_minimum_v2, shd_mean))

    return shade_list, npv_list, gv_list

def init_weights(shape):
    # initialize the parameter for the net
    weight = tf.Variable(tf.random_normal(shape, stddev=0.01)) #, seed=42
    return weight

def en_net(X, w_in, beta, gama, alpha):
    l1 = tf.matmul(X, w_in)
    mean, var = tf.nn.moments(l1, axes=0, keep_dims=True)
    l1 = tf.nn.batch_normalization(l1, mean, var, beta, gama, epsilon)  # 我的BN没有滑动平均 和BN的最原始的文章不一样
    l2 = tf.nn.relu(features=l1 - alpha) #
    # l2 = tf.nn.sigmoid(l1 - alpha) # 0.58
    # l2 = tf.nn.leaky_relu(l1 - alpha)
    # l2 = tf.matmul(l1,w_2)
    # mean2, var2 = tf.nn.moments(l2, axes=0, keep_dims=True)
    # l2 = tf.nn.batch_normalization(l2, mean2, var2, beta2, gama2, epsilon)  # 我的BN没有滑动平均 和BN的最原始的文章不一样
    # l2 = tf.nn.relu(l1 - alpha2)
    result = tf.nn.softmax(l2)
    # result = nor(l1)

    return l2, result

def de_net(X, w_out):
    out = tf.matmul(X, w_out)
    return out

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

    # set the range of the endmembers
    gv_range = endmember_list[2]
    npv_range = endmember_list[1]
    shd_range = endmember_list[0]

    # transform the image for training
    Img = Img.astype(np.float32)
    Img_clean = Img_clean.astype(np.float32)
    num_of_pixels, in_size = Img_clean.shape

    # setting the parameter for the autoencoder
    # initialize the weight to obtain the feature abstract
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

    # initial_w_out = Ed_tmp_initial.transpose()
    if pre_initial:
        # if shade_initial[1] < ever_initial[1] <= dec_initial[1]*1.1:
        initial_w_out = Ed_tmp_initial.transpose()
        w_out = tf.Variable(initial_value=initial_w_out.astype(np.float32))  # endnum, bands
        # w_in = tf.Variable(initial_value=initial_w_out.transpose().astype(np.float32))
        w_in = init_weights([in_size, endnum])
        # else:
        #     print('\nIndex of the initial endmember is not suitable!\n')
        #     initial_w_out = init_weights([endnum, in_size])
        #     w_out = init_weights([endnum, in_size])  # initialized endmember: randomly or pre-initialized
        #     w_in = init_weights([in_size, endnum])
    else:
        initial_w_out = init_weights([endnum, in_size])
        w_out = init_weights([endnum, in_size])  # initialized endmember: randomly or pre-initialized
        w_in = init_weights([in_size, endnum])

    initial_abu = get_abundance_map(Img_clean, Ed_tmp_initial)
    initial_abu = initial_abu.reshape(-1, 3)  # [mask_ind]

    beta = tf.Variable(tf.zeros([1]))
    gama = tf.Variable(tf.ones([1]))
    alpha = tf.Variable(tf.zeros([1, endnum]))
    # autoencoder parameter list
    theta_ae = [w_in, w_out, beta, gama, alpha]  #

    # parameter for dis_net : discriminator
    # D01 = init_weights([in_size, endnum])
    # D12 = init_weights([endnum, 1])

    # theta_D = [D01, D12]

    # saver
    end_epo = []
    end_all = []

    # define the placeholder for the input and the output
    X = tf.placeholder(tf.float32, [None, in_size])
    Y = tf.placeholder(tf.float32, [None, in_size])
    X_test = tf.placeholder(tf.float32, [None, in_size])  # for predicting whole abundacne map


    # training process
    z, en_result = en_net(X, w_in, beta, gama, alpha)
    de_result = de_net(en_result, w_out)

    # discriminator training
    # d2_real, real = dis_net(X, D01, D12)
    # d2_fake, fake = dis_net(de_result, D01, D12)

    _, en_result_1 = en_net(X_test, w_in, beta, gama, alpha)
    de_result_2 = de_net(en_result_1, w_out)

    # define the loss function for discriminator
    # mse2 = tf.losses.mean_squared_error(d2_real, d2_fake)
    # D_loss = -tf.reduce_mean(real) + tf.reduce_mean(fake)
    # define the loss function and the optimizer
    ae_loss = tf.losses.mean_squared_error(Y, de_result) #+ 0.5*mse2
    # ae_loss = tf.losses.absolute_difference(Y, de_result)

    # D_lr = tf.placeholder(tf.float32)
    AE_lr = tf.placeholder(tf.float32)  # create the graph for the learning late
    # ae_solver = tf.train.AdamOptimizer(learning_rate=AE_lr).minimize(ae_loss, var_list=theta_ae) # ,
    ae_solver = tf.train.AdadeltaOptimizer(learning_rate=AE_lr).minimize(ae_loss, var_list=theta_ae) # ,
    # ae_solver = tf.train.AdagradOptimizer(learning_rate=AE_lr).minimize(ae_loss, var_list=theta_ae)
    # ae_solver = tf.train.GradientDescentOptimizer(learning_rate=AE_lr).minimize(ae_loss)#, var_list=theta_ae
    # D_solver = tf.train.AdamOptimizer(learning_rate=D_lr).minimize(D_loss, var_list=theta_D) #GradientDescentOptimizer
    # Ecological Constraints
    # reback_w_out = tf.assign(w_out, tf.cast(tf.Variable(initial_w_out), tf.float32))

    # reback_w_in = tf.assign(w_in, init_weights([in_size, endnum]))


    # non-negative constraint
    clip_op_all = tf.assign(w_out, tf.clip_by_value(w_out, 0, gv_range[0].max()))

    if begin_with_zero:
        # limit the range of evergreen
        clip_op_0 = tf.assign(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[0], :],  # ,3
                              tf.clip_by_value(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[0], :], 0,  # gv_range[1, :]
                                               gv_range[0, :]))
        # limit the range of deciduous
        clip_op_1 = tf.assign(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[1], :],
                              tf.clip_by_value(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[1], :], 0,  # npv_range[1, :]
                                               npv_range[0, :]))
        # limit the range of shade
        clip_op_2 = tf.assign(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[2], :],
                              tf.clip_by_value(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[2], :], 0,  # shd_range[1, :]
                                               npv_range[1, :]))  # shd_range[0, :]
    else:
        clip_op_0 = tf.assign(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[0], :],  # ,3
                              tf.clip_by_value(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[0], :], gv_range[1, :],  #
                                               gv_range[0, :]))
        # limit the range of deciduous
        clip_op_1 = tf.assign(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[1], :],
                              tf.clip_by_value(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[1], :], npv_range[1, :],  #
                                               npv_range[0, :]))
        # limit the range of shade
        clip_op_2 = tf.assign(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[2], :],
                              tf.clip_by_value(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[2], :], shd_range[1, :],  #
                                               shd_range[0, :]))  #
    #
    # clip_op_1_nir = tf.assign(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[1], 3],
    #                           tf.clip_by_value(w_out[tf.nn.top_k(w_out[:, 3], k=3).indices[1], 3], npv_range[1, 3],  #
    #                                            npv_range[0, 3]))

    img_mse = tf.losses.mean_squared_error(X_test, de_result_2)

    # training the whole auto-encoder
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # plot_1(w_out)
        ae_lossforplt = []
        loss_all = []
        # best_loss for early stopping
        best_loss = 1
        not_decrease_record = 0
        last_improvement = 0

        index_list = np.arange(num_of_pixels)

        np.random.seed(42)
        np.random.shuffle(index_list)

        # corrupt = sess.run(tf.nn.dropout(Img, keep_prob=0.25)) # dropout
        # R = corrupt[index_list]
        R = Img[index_list]
        R_clean = Img_clean[index_list]

        eco_step = 0

        for step in range(epoch):

            for i in range(num_of_pixels // batch_size + 1):
                # Prepare Data
                if i < num_of_pixels // batch_size:
                    # batch_x = data[i * batch_size:(i + 1) * batch_size, :]
                    batch_x = R[i * batch_size:(i * batch_size + batch_size)]
                    batch_y = R_clean[i * batch_size:(i * batch_size + batch_size)]
                else:
                    # batch_x = data[i * batch_size:, :]
                    batch_x = R[i * batch_size:]
                    batch_y = R_clean[i * batch_size:]

                ae_solver_, ae_loss_ = sess.run([ae_solver, ae_loss],
                                                feed_dict={X: batch_x, AE_lr: learning_rate, Y: batch_y})

                # _, d_loss = sess.run([D_solver, D_loss], feed_dict={X: batch_x, D_lr: learning_rate*0.01})

                sess.run(clip_op_all)
                sess.run(clip_op_0)
                sess.run(clip_op_1)
                # if shade_initial[1] < ever_initial[1] <= dec_initial[1] * 1.1:
                #     sess.run(clip_op_1)
                # else:
                #     sess.run(clip_op_1_nir)
                sess.run(clip_op_2)

            # ae_lossforplt.append(ae_loss_)  # find out the best model
            if verbose:
                print('Step: {}'.format(step))
                print('AE_loss: {:.5}'.format(ae_loss_))
                # print('AutoLearning rate: %f' % (sess.run(ae_solver_._lr)))
                print('Learning rate: {:.5}'.format(learning_rate))
            # After training, obtain the endmember and abundance map
            # 估计的丰度矩阵en_result_, 端元矩阵w_out_
            # en_result_, w_out_ = sess.run([en_result_1, w_out], feed_dict={X_test: r})  # predict the whole image
            en_result_, w_out_, final_loss_auto = sess.run([en_result_1, w_out, img_mse],
                                                           feed_dict={X_test: Img_clean})  # predict the whole image

            # final_loss_auto = ae_loss_ # just using the training loss value -- Not suitable!!
            if verbose:
                print('Image Loss Value :', final_loss_auto)
            # print('Initial Image Loss Value:', final_loss_initial)
            # ae_lossforplt.append(final_loss_auto)  # save the loss value and find out the best model
            # final_loss_auto = ae_loss_ # just using the training loss value -- Not suitable!!
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

            ae_lossforplt.append(final_loss_auto)
            end_epo.append(w_out_)

            eco_step += 1

            # if ((shade_[0] <= ever_[0] < dec_[0]) and (shade_[1] < ever_[1] < dec_[1]) and (
            #     shade_[2] <= ever_[2] < dec_[2])):
            # ae_lossforplt.append(final_loss_auto)
            # end_epo.append(w_out_)

            # if shade_[1] < ever_[1] <= dec_[1]*1.10:
            #     print('Match Ecological Meaning')
            #     ae_lossforplt.append(final_loss_auto)
            #     end_epo.append(w_out_)
            #     eco_step += 1
            # else:
            #     print('Not match ecological meaning, reinitialize the parameter!!')
            #     eco_step = 0
            #     continue
                # with tf.Session() as sess2:
                #     sess2.run(tf.global_variables_initializer())
                #     sess2.run(reback_w_out)
                #     sess2.run(reback_w_in)
                # # if shade_initial[1] + 0.005 < ever_initial[1] < dec_initial[1]:
                # # if shade_initial[1] < ever_initial[1] <= dec_initial[1]:
                # #     print('Index match')
                #     # with tf.Session() as sess2:
                #     #     sess2.run(tf.global_variables_initializer())
                #     #     sess2.run(reback_w_out)
                #     # w_out = tf.Variable(initial_value=initial_w_out.astype(np.float32))
                # continue

            if step > 0:
                if ae_loss_ < loss_all[step - 1]:
                    if ae_loss_ < best_loss - delta_increasing:  # default 0.5e-5
                        not_decrease_record = 0
                        best_loss = ae_loss_
                        last_improvement = 0
                        if verbose:
                            print('\nBest loss value: ', best_loss, '\n')
                    else:
                        last_improvement += 1

                    if eco_step > 10:
                        if last_improvement > require_improvement - 1:# and eco_step > 5:
                            # if learning_rate <= 1e-5:
                            if verbose:
                                print('Early Stop')
                            break
                else:
                    if verbose:
                        print('loss value do not decrease')
                    break
                #     not_decrease_record += 1
                # if not_decrease_record > 2:
                #     print('Not decreasing more than 2 times')
                #     break
            # else:
            #     not_decrease_record = 0

    tf.keras.backend.clear_session()

    # get the final endmembers and the abundance map
    if len(ae_lossforplt) != 0:
        min_index = ae_lossforplt.index(min(ae_lossforplt))  # maybe overfitting in this way!
        final_end = end_epo[min_index].transpose()
    else:
        if (shade_initial[1] <= ever_initial[1] < dec_initial[1]*1.10):

            final_end = Ed_tmp_initial.transpose()

        else:
            constant_w_out = tf.placeholder(tf.float32, shape=(initial_w_out.shape[0], initial_w_out.shape[1]))
            tf_img = tf.placeholder(tf.float32, shape=(Img_clean.shape[0], Img_clean.shape[1]))
            tf_abu_init = tf.placeholder(tf.float32, shape=(initial_abu.shape[0], initial_abu.shape[1]))
            de_result_initial = tf.matmul(tf_abu_init, constant_w_out)
            # final_loss_inital = K.mean(K.square(de_result_initial - tf_img))
            # final_loss_initial_ = normSAD(tf_img, de_result_initial)
            final_loss_initial_ = tf.losses.mean_squared_error(tf_img, de_result_initial)
            # final_loss_initial_ = SAD(tf_img, de_result_initial)
            with tf.Session() as sess2:
                sess2.run(tf.global_variables_initializer())
                final_loss_initial = sess2.run(final_loss_initial_, feed_dict={tf_img: Img_clean.astype(np.float32),
                                                                               tf_abu_init: initial_abu,
                                                                               constant_w_out: Ed_tmp_initial.transpose()})


            min_index = loss_all.index(min(loss_all))  # maybe overfitting in this way!
            final_end = end_all[min_index].transpose()

            if (loss_all[min_index] > final_loss_initial):
                final_end = initial_w_out.transpose()


    # re-rank the endmember
    new_rank_index = final_end[3, :].argsort()  # shade, deciduous, evergreen
    final_end = final_end[:, new_rank_index]

    # compare with the index constrain
    shade_ = final_end[:, 0]
    dec_ = final_end[:, 1]
    ever_ = final_end[:, 2]
    if verbose:
        print('final_end_autoencoder:')
        print('Shade: ', shade_[[0, 1, 2, 3]])
        print('Dec: ', dec_[[0, 1, 2, 3]])
        print('Ever: ', ever_[[0, 1, 2, 3]])
        # Initial endmember
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
    # using the linear spectral unmixing to obtain the abundance map

    mask_ind = np.all(input_img != 0, axis=1)

    new_input_img = input_img[mask_ind].transpose([1, 0])

    img_abundance = unmixing.sunsal(endmember_mean, new_input_img, al_iters=200,
                                       positivity=True, addone=True, verbose=False)[0].transpose([1, 0])

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
