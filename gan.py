import os

os.environ['PYTHONHASHSEED'] = '0'

from tensorflow import set_random_seed as tf_set_random_seed

tf_set_random_seed(1234)

import numpy as np

np.random.seed(42)

import random

random.seed(7)

# Can turn off GPU on CPU-only machines; maybe results in faster startup.
use_GPU = True
if use_GPU is False:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from constants import *
from gan_plots import *
from dense_nn import *

import tensorflow as tf
import numpy as np
import pickle
import math
from imageio import imread, mimsave
from functools import partial




class VIXMatHandler:
    def __init__(self, vix_mat):
        self.vix_mat_shape = vix_mat.shape

        self.vix_mat_mean = np.mean(vix_mat)
        self.vix_mat_std = np.std(vix_mat)

        self.vix_mat_stdz = (vix_mat - self.vix_mat_mean) / self.vix_mat_std

        vix_mat_ln_orig = np.log(vix_mat)
        self.vix_max_ln_mean = np.mean(vix_mat_ln_orig)
        self.vix_mat_ln = vix_mat_ln_orig - self.vix_max_ln_mean

        self.vix_mat_total = np.concatenate([self.vix_mat_stdz, self.vix_mat_ln], axis=1)

    def simulated_vix(self, center=None, dispersion=None, shape=None):
        if center is None:
            center = self.vix_mat_mean
        if dispersion is None:
            dispersion = self.vix_mat_std
        if shape is None:
            shape = self.vix_mat_shape


        rnd = np.random.normal(loc=center, scale=dispersion, size=shape)

        max_val = center + dispersion
        rnd[rnd > max_val] = max_val

        min_val = center - dispersion
        rnd[rnd < min_val] = min_val

        rnd_stdz = (rnd - self.vix_mat_mean) / self.vix_mat_std

        rnd_ln_orig = np.log(rnd)
        rnd_ln = rnd_ln_orig - self.vix_max_ln_mean

        return np.concatenate([rnd_stdz, rnd_ln], axis=1)


class Batches:
    def __init__(self, data):
        self.n = data.shape[0]
        self.n_batches = math.ceil(self.n / BATCH_SIZE)
        self.shuffle()

    def shuffle(self):
        self.idx = np.arange(0, self.n)
        np.random.shuffle(self.idx)

        self.index_list = list()

        if self.n <= BATCH_SIZE:
            self.index_list.append(self.idx)
        else:
            for i in range(self.n_batches):
                start = i * BATCH_SIZE
                stop = min(self.n, (i + 1) * BATCH_SIZE)
                self.index_list.append(self.idx[start:stop])


def generate_random(shape, epoch):
    rand = np.random.normal(loc=RAND_MEAN, scale=RAND_STD, size=shape)
    if epoch >= EPOCHS:
        pass
    else:
        iter_scale = (1.0 - (float(epoch) / float(EPOCHS)))
        rand_skew = randn_skew_fast(shape, alpha=random.uniform(-3., 3.), loc=random.uniform(-.5, .5),
                                    scale=random.uniform(0.5, 3.))
        rand = rand + rand_skew * iter_scale
    return rand


def add_random(mat, epoch):
    if epoch >= EPOCHS:
        return mat
    else:
        iter_scale = (1.0 - (float(epoch) / float(EPOCHS)))

        rand = randn_skew_fast(mat.shape, alpha=random.uniform(-3., 3.), loc=random.uniform(-.5, .5),
                               scale=random.uniform(0.5, 3.))
        adj = DISCRIMINATOR_RANDOM_SCHEDULE * iter_scale * rand
        return mat + adj


def sample_matrix(x, vix, draw_correl_charts=False):
    x_len = x.shape[0]
    num_rows = x_len - SAMPLE_LEN + 1

    x_mat = np.zeros((num_rows, N_SERIES, SAMPLE_LEN))
    vix_mat = np.zeros((num_rows, N_CONDITIONALS, SAMPLE_LEN))

    correls = np.zeros((num_rows, N_SERIES))

    for n in range(num_rows):
        col0 = x[n:(n + SAMPLE_LEN), 0]
        col1 = x[n:(n + SAMPLE_LEN), 1]
        col2 = x[n:(n + SAMPLE_LEN), 2]

        vix_sample = vix[n:(n + SAMPLE_LEN)]

        correls[n, 0] = np.corrcoef(col0, col1)[0, 1]
        correls[n, 1] = np.corrcoef(col0, col2)[0, 1]
        correls[n, 2] = np.corrcoef(col1, col2)[0, 1]

        x_mat[n, 0, :] = col0
        x_mat[n, 1, :] = col1
        x_mat[n, 2, :] = col2

        vix_mat[n, 0, :] = vix_sample

    if draw_correl_charts:
        dir = ['../images/base_correlations/']
        delete_files_in_folder(dir[0])
        hist_elements = [{'data': correls[:, 0], 'label': 'R3000 vs. AAA'},
                         {'data': correls[:, 1], 'label': 'R3000 vs. EM HY'},
                         {'data': correls[:, 2], 'label': 'AAA vs. EM HY'}]
        f_name = f'correls_actual.png'
        dist_chart(hist_elements, 'Correlations', 'Frequency', 'Actual data correlations (1999 to 2018)', f_name=f_name,
                   bins=30, directories=dir, scaleX=[-1., 1.])

    return x_mat, correls, vix_mat


def randn_skew_fast(shape, alpha=0.0, loc=0.0, scale=1.0):
    sigma = alpha / np.sqrt(1.0 + alpha ** 2)
    u0 = np.random.randn(shape[0], shape[1], shape[2])
    v = np.random.randn(shape[0], shape[1], shape[2])
    u1 = (sigma * u0 + np.sqrt(1.0 - sigma ** 2) * v) * scale
    u1[u0 < 0] *= -1
    u1 = u1 + loc
    return u1


def load_data():
    if USE_SAVED_X:
        x = pickle.load(open(f'{X_SAVE_PATH}last_x.p', 'rb'))
        x_mat = pickle.load(open(f'{X_SAVE_PATH}last_x_mat.p', 'rb'))
        correls_actual = pickle.load(open(f'{X_SAVE_PATH}last_correls_actual.p', 'rb'))
        vix = pickle.load(open(f'{X_SAVE_PATH}last_vix.p', 'rb'))
        vix_mat = pickle.load(open(f'{X_SAVE_PATH}last_vix_mat.p', 'rb'))
    else:
        from weekly_returns import weekly_returns as x
        from weekly_returns import vix
        x_mat, correls_actual, vix_mat = sample_matrix(x, vix, draw_correl_charts=True)

        pickle.dump(x, open(f'{X_SAVE_PATH}last_x.p', 'wb'))
        pickle.dump(x_mat, open(f'{X_SAVE_PATH}last_x_mat.p', 'wb'))
        pickle.dump(correls_actual, open(f'{X_SAVE_PATH}last_correls_actual.p', 'wb'))
        pickle.dump(vix, open(f'{X_SAVE_PATH}last_vix.p', 'wb'))
        pickle.dump(vix_mat, open(f'{X_SAVE_PATH}last_vix_mat.p', 'wb'))

    return x, x_mat, correls_actual, vix, vix_mat


def main(saved_model_dir=None):
    x, x_mat, correls_actual, vix, vix_mat = load_data()

    vix_handler = VIXMatHandler(vix_mat)

    x_mat_mean = np.mean(x_mat)
    x_mat_std = np.std(x_mat)
    x_mat_normalized = (x_mat - x_mat_mean) / x_mat_std

    x_mat_split = np.split(x_mat, indices_or_sections=N_SERIES, axis=1)
    x_mat_means = list()
    x_mat_stdevs = list()

    for m in x_mat_split:
        mm = np.squeeze(m, 1)
        x_mat_means.append(np.mean(mm, axis=1) * 100.)
        x_mat_stdevs.append(np.std(mm, axis=1) * 100.)

    tf.reset_default_graph()

    if DISCRIMINATOR_CONVOLUTION:
        from conv_nn import create_cnn_disc_graph
        discriminator = partial(create_cnn_disc_graph)
        generator = partial(create_dense_gen_graph, reuse=False)
    elif DISCRIMINATOR_RNN:
        from rnn import create_rnn_disc_graph, create_rnn_gen_graph

        discriminator = partial(create_rnn_disc_graph)
        generator = partial(create_rnn_gen_graph, reuse=False)
    else:
        discriminator = partial(create_dense_disc_graph)
        generator = partial(create_dense_gen_graph, reuse=False)

    X = tf.placeholder(tf.float32, [None, N_SERIES, SAMPLE_LEN])
    Z = tf.placeholder(tf.float32, [None, N_SERIES, SAMPLE_LEN])
    C = tf.placeholder(tf.float32, [None, N_CONDITIONALS, SAMPLE_LEN])

    g_out, g_train = generator(Z=Z, conditionals=C)
    x_out, x_train = discriminator(X=X, conditionals=C, reuse=False)
    z_out, z_train = discriminator(X=g_out, conditionals=C, reuse=True)

    disc_loss_base = -tf.reduce_mean(x_out) + tf.reduce_mean(z_out)
    gen_loss_base = -tf.reduce_mean(z_out)

    scale = 10.
    epsilon = tf.random_uniform([], minval=0., maxval=1.)
    x_h = epsilon * X + (1. - epsilon) * g_out

    disc_for_grad, training_flag_for_grad = discriminator(X=x_h, conditionals=C, reuse=True)
    grad_d_x_h = tf.gradients(disc_for_grad, x_h)[0]

    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad_d_x_h), axis=1))
    grad_pen = tf.reduce_mean(tf.square(grad_norm - 1.))

    disc_loss_base += scale * grad_pen

    DISCRIMINATOR_TRAINS_PER_BATCH = 5 * GENERATOR_TRAINS_PER_BATCH

    if L2_REGULARIZATION > 0.:
        gen_l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=GENERATOR_SCOPE)
        disc_l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=DISCRIMINATOR_SCOPE)

        gen_loss = tf.add_n([gen_loss_base] + gen_l2_loss)
        disc_loss = tf.add_n([disc_loss_base] + disc_l2_loss)
    else:
        gen_loss = gen_loss_base
        disc_loss = disc_loss_base

    if GENERATOR_OPTIMIZER == OptimizerType.ADAM:
        gen_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
    elif GENERATOR_OPTIMIZER == OptimizerType.RMSPROP:
        gen_opt = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.0, decay=0.9, epsilon=1e-10)
    else:
        raise ValueError('Unknown optimizer selection for generator (choose Adam or RMSProp.')

    if DISCRIMINATOR_OPTIMIZER == OptimizerType.ADAM:
        disc_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
    elif DISCRIMINATOR_OPTIMIZER == OptimizerType.RMSPROP:
        disc_opt = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.0, decay=0.9, epsilon=1e-10)
    else:
        raise ValueError('Unknown optimizer selection for generator (choose Adam or RMSProp.')

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=GENERATOR_SCOPE)
    gen_step = gen_opt.minimize(gen_loss, var_list=gen_vars)

    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=DISCRIMINATOR_SCOPE)
    disc_step = disc_opt.minimize(disc_loss, var_list=disc_vars)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    save_dir_all = "../model_saves/all/"

    png_files_01 = list()
    png_files_02 = list()
    png_files_12 = list()

    png_moments_files = [list(), list(), list()]

    png_points_files = [list(), list(), list()]

    generated_values = list()

    with tf.Session() as sess:
        if saved_model_dir is not None:
            saver.restore(sess, saved_model_dir)

            dir_name = f'{RESULTS_USED_DIR}simulated_results/'

            rnd = np.random.normal(loc=RAND_MEAN, scale=RAND_STD, size=x_mat_normalized.shape)

            kp_prob = KEEP_PROB if GENERATOR_DROPOUT_ALWAYS_ON else 1.
            index_to_plot = 0
            series_to_plot = 948

            vols_set = [((24., 2.), (18., 2.)), ((16., 2.), (14., 2.)), ((14., 2.), (10., 2.)), ((14., 2.), (10., 2.)),
                        ((8., 2.), (6., 2.)), ((8., 2.), (6., 2.)), ((24., 2.), (12., 2.)), ((20., 2.), (10., 2.)),
                        ((40., 2.), (12., 2.)), ((48., 2.), (12., 2.)), ((51.095, 28.04), (14.005, 3.74))]

            for vi, vols in enumerate(vols_set):

                data_sets = list()

                for v in vols:
                    rnd_vol = vix_handler.simulated_vix(center=v[0], dispersion=v[1])

                    genval = sess.run(g_out, feed_dict={
                        Z: rnd[:series_to_plot], C: rnd_vol[:series_to_plot], g_train: kp_prob, x_train: kp_prob,
                        z_train: kp_prob
                        })

                    genval_unnormalized = (genval * x_mat_std) + x_mat_mean

                    generated_to_plot = genval_unnormalized[:, index_to_plot, :]
                    data_sets.append(generated_to_plot)

                    labels = ['R3000', 'AAA', 'EMHY']
                    for s in range(N_SERIES):
                        data = genval_unnormalized[:, s, :]
                        f_name = f'{labels[s]}_simulated_data_{int(v[0])}vol_{int(v[1])}volStd.csv'
                        np.savetxt(dir_name + f_name, data, delimiter=',')

                legend_loc = 2
                labels = ['High volatility', 'Low volatility']
                # legend_loc = 3
                # labels=['GFC VIX (centered at 51)', '2005 VIX (centered at 14)'],
                f_name = f'simulated_vol{int(vols[0][0])}_{int(vols[0][1])}_vol{int(vols[1][0])}_{int(vols[1][1])}.png'
                cumul_return_plot(data_sets=data_sets, f_name=f_name, dir_name=dir_name,
                                  legend_loc=legend_loc, labels=labels)
            return None

        init.run()

        for e in range(EPOCHS):
            epch = e + 1

            if epch > 1:
                # The generator gets a new set of random data every epoch.
                # This is because one motivation for a GAN is to generate an unlimited number of samples.
                rnd = generate_random(x_mat_normalized.shape, epch)

                Z_batch = Batches(rnd)
                X_batch = Batches(x_mat_normalized)

                assert Z_batch.n_batches == X_batch.n_batches
                n_batches = Z_batch.n_batches

                g_losses = np.zeros((n_batches, 1))
                d_losses = np.zeros((n_batches, 1))

                for b in range(n_batches):
                    Z_in = rnd[Z_batch.index_list[b]]

                    xc_idx = X_batch.index_list[b]
                    X_in = x_mat_normalized[xc_idx]
                    C_in = vix_handler.vix_mat_total[xc_idx]

                    for t in range(DISCRIMINATOR_TRAINS_PER_BATCH):
                        if DISCRIMINATOR_RNN:
                            _, dloss = sess.run([disc_step, disc_loss], feed_dict={
                                X: add_random(X_in, epch), Z: Z_in, C: C_in, g_train: KEEP_PROB, x_train: KEEP_PROB,
                                z_train: KEEP_PROB
                                })
                        else:
                            _, dloss = sess.run([disc_step, disc_loss], feed_dict={
                                X: add_random(X_in, epch), Z: Z_in, C: C_in, g_train: True, x_train: True, z_train: True
                                })

                    for t in range(GENERATOR_TRAINS_PER_BATCH):
                        if DISCRIMINATOR_RNN:
                            _, gloss = sess.run([gen_step, gen_loss], feed_dict={
                                Z: Z_in, C: C_in, g_train: KEEP_PROB, x_train: KEEP_PROB, z_train: KEEP_PROB
                                })
                        else:
                            _, gloss = sess.run([gen_step, gen_loss], feed_dict={
                                Z: Z_in, C: C_in, g_train: True, x_train: True, z_train: True
                                })

                    g_losses[b] = gloss
                    d_losses[b] = dloss

                print(f"Iter: {epch}   Disc_loss: {int(np.mean(dloss))}   Gen_loss: {int(np.mean(gloss))}")

            # save fit plot every several epochs
            if epch == 1 or epch % 5 == 0 or epch == (EPOCHS + 1):

                if epch >= MIN_EPOCHS:
                    saver.save(sess, f"{save_dir_all}epoch_{epch}/model.ckpt")

                rnd_validation = np.random.normal(loc=RAND_MEAN, scale=RAND_STD, size=x_mat_normalized.shape)

                if DISCRIMINATOR_RNN:
                    kp_prob = KEEP_PROB if GENERATOR_DROPOUT_ALWAYS_ON else 1.
                    genval = sess.run(g_out, feed_dict={
                        Z: rnd_validation, C: vix_handler.vix_mat_total, g_train: kp_prob, x_train: kp_prob,
                        z_train: kp_prob
                        })
                else:
                    genval = sess.run(g_out, feed_dict={
                        Z: rnd_validation, C: vix_handler.vix_mat_total, g_train: False, x_train: False, z_train: False
                        })

                # plot distributions
                genval_unnormalized = (genval * x_mat_std) + x_mat_mean
                genval_unnormalized_split_temp = np.split(genval_unnormalized, indices_or_sections=N_SERIES, axis=1)

                genval_unnormalized_split = list()
                for gust in genval_unnormalized_split_temp:
                    genval_unnormalized_split.append(np.squeeze(gust, axis=1))

                f_names = ['R3000', 'AAA', 'EMHY']
                title_names = ['Russell 3000', 'AAA', 'Emerging markets high yield']

                size = 80

                # plot mean/stdev
                gen_unnrml_series_means = list()
                gen_unnrml_series_stdevs = list()

                for g in genval_unnormalized_split:
                    gen_unnrml_series_means.append(np.mean(g, axis=1) * 100.)
                    gen_unnrml_series_stdevs.append(np.std(g, axis=1) * 100.)

                directories = ['../images/pngs_moments/', '../images/pngs_moments_for_gif/']

                colors = ('#00BFFFff', '#c32148ff', '#fd5e0fff', '#228b22ff', '#daa520ff', '#b710aaff')
                for s in range(N_SERIES):
                    elements = [{
                        'x': x_mat_means[s], 'y': x_mat_stdevs[s], 'label': 'actual', 'alpha': 1.,
                        'color': colors[s * 2], 'size': size
                        }, {
                        'x': gen_unnrml_series_means[s], 'y': gen_unnrml_series_stdevs[s], 'label': 'generated',
                        'alpha': 1., 'color': colors[s * 2 + 1], 'size': size
                        }]

                    file_name = f'{f_names[s]}_epoch{epch}.png'
                    scatter(elements, 'Weekly mean return (%)', 'Weekly standard deviation (%)',
                            f'{title_names[s]}: actual vs. generated dispersion (epoch {epch})', save_file=file_name,
                            directories=directories)

                    png_moments_files[s].append(file_name)

                # plot points
                directories = ['../images/pngs_distributions/', '../images/pngs_distributions_for_gif/']
                for s in range(N_SERIES):
                    elements = [{'data': x_mat_split[s].flatten() * 100., 'label': f'actual'},
                                {'data': genval_unnormalized_split[s].flatten() * 100., 'label': f'generated'}]

                    file_name = f'{f_names[s]}_epoch{epch}.png'
                    title = f'{title_names[s]}: actual vs. generated data (epoch {epch})'
                    dist_chart(elements, 'Weekly returns (%)', 'Frequency', title=title, f_name=file_name, bins=100,
                               color_start=s * 2, directories=directories)

                    png_points_files[s].append(file_name)

                # plot correlations
                gen_series_temp = np.split(genval, indices_or_sections=N_SERIES, axis=1)

                gen_series = list()
                for gst in gen_series_temp:
                    gen_series.append(np.squeeze(gst, axis=1))

                nr = gen_series[0].shape[0]

                corr_01 = np.zeros((nr, 1))
                corr_02 = np.zeros((nr, 1))
                corr_12 = np.zeros((nr, 1))

                for r in range(nr):
                    corr_01[r, 0] = np.corrcoef(gen_series[0][r], gen_series[1][r])[0, 1]
                    corr_02[r, 0] = np.corrcoef(gen_series[0][r], gen_series[2][r])[0, 1]
                    corr_12[r, 0] = np.corrcoef(gen_series[1][r], gen_series[2][r])[0, 1]

                generated_values.append({
                    'epoch': epch, 'genval': genval, 'corr_01': corr_01, 'corr_02': corr_02, 'corr_12': corr_12
                    })

                x_lab = 'Correlations'
                y_lab = 'Frequency'
                bins = 30
                directories = ['../images/pngs_correls/', '../images/pngs_correls_for_gif/']
                scaleX = [-1., 1.]

                title = f'Russell 3000 vs. AAA: actual and generated correlations (epoch {epch})'
                f_name_01 = f'R3000vAAA_epoch{epch}.png'
                el = [{'data': correls_actual[:, 0], 'label': 'actual'}, {'data': corr_01, 'label': 'generated'}]
                dist_chart(el, x_lab, y_lab, title, f_name_01, bins=bins, color_start=0, directories=directories,
                           scaleX=scaleX)

                title = f'Russell 3000 vs. Emerging markets HY: actual and generated correlations (epoch {epch})'
                f_name_02 = f'R3000vEMHY_epoch{epch}.png'
                el = [{'data': correls_actual[:, 1], 'label': 'actual'}, {'data': corr_02, 'label': 'generated'}]
                dist_chart(el, x_lab, y_lab, title, f_name_02, bins=bins, color_start=2, directories=directories,
                           scaleX=scaleX)

                title = f'AAA vs. Emerging markets HY: actual and generated correlations (epoch {epch})'
                f_name_12 = f'AAAvEMHY_epoch{epch}.png'
                el = [{'data': correls_actual[:, 2], 'label': 'actual'}, {'data': corr_12, 'label': 'generated'}]
                dist_chart(el, x_lab, y_lab, title, f_name_12, bins=bins, color_start=4, directories=directories,
                           scaleX=scaleX)

                png_files_01.append(f_name_01)
                png_files_02.append(f_name_02)
                png_files_12.append(f_name_12)

    gif_names = ['01', '02', '12']
    gif_vars = [png_files_01, png_files_02, png_files_12]

    for i, gv in enumerate(gif_vars):
        images_dist_plot = list()
        for f in gv:
            img = imread('../images/pngs_correls_for_gif/' + f)
            images_dist_plot.append(img)
        mimsave(f'../images/gifs/correlations_{gif_names[i]}.gif', images_dist_plot, duration=1.)

    gif_names = ['R3000', 'AAA', 'EMHY']
    for i, dist in enumerate(png_moments_files):
        images_dist_plot = list()
        for f in dist:
            img = imread('../images/pngs_moments_for_gif/' + f)
            images_dist_plot.append(img)
        mimsave(f'../images/gifs/moments_{gif_names[i]}.gif', images_dist_plot, duration=1.)

    for i, dist in enumerate(png_points_files):
        images_dist_plot = list()
        for f in dist:
            img = imread('../images/pngs_distributions_for_gif/' + f)
            images_dist_plot.append(img)
        mimsave(f'../images/gifs/distributions_{gif_names[i]}.gif', images_dist_plot, duration=1.)


if __name__ == '__main__':
    delete_files_in_folder('../model_saves/all/')
    delete_files_in_folder('../images/pngs_correls/')
    delete_files_in_folder('../images/pngs_correls_for_gif/')
    delete_files_in_folder('../images/gifs/')
    delete_files_in_folder('../images/pngs_moments/')
    delete_files_in_folder('../images/pngs_moments_for_gif/')
    delete_files_in_folder('../images/pngs_distributions/')
    delete_files_in_folder('../images/pngs_distributions_for_gif/')

    # saved_model_dir = None
    saved_model_dir = f'{RESULTS_USED_DIR}model_saves/all/epoch_100/model.ckpt'
    main(saved_model_dir=saved_model_dir)
