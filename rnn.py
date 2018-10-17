from constants import *

import tensorflow as tf
import numpy as np
from functools import partial

XAV_INIT = tf.contrib.layers.xavier_initializer()
VS_INIT = tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='normal')
RND_INIT = tf.random_normal_initializer()


def create_rnn_gen_graph(Z, conditionals, reuse=False):
    with tf.variable_scope(GENERATOR_SCOPE, reuse=reuse):

        with tf.name_scope("generator_input"):

            keep_prob = tf.placeholder_with_default(KEEP_PROB, shape=())

            Z_all = tf.concat([Z, conditionals], axis=1)

            Z_sequences = tf.unstack(tf.transpose(Z_all, perm=[2, 0, 1]))

        with tf.name_scope("generator_rnn"):
            RNN_cell = partial(tf.contrib.rnn.GRUCell, activation=tf.nn.tanh, reuse=reuse, kernel_initializer=XAV_INIT)

            cells = [RNN_cell(num_units=neurons) for neurons in RNN_NEURONS]

            cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob) for cell in cells]

            multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)

            rnn_outputs, states = tf.contrib.rnn.static_rnn(multi_layer_cell, Z_sequences, dtype=tf.float32)

        with tf.name_scope("generator_intermediate_dense"):
            dense_varscope = 'RNN_GEN_TimeDistributed'
            dense_out = list()
            for r, rnn_out in enumerate(rnn_outputs):
                dense_reuse = False if (reuse is False and r == 0) else True
                with tf.variable_scope(dense_varscope, reuse=dense_reuse):
                    dense_mid = tf.layers.dense(rnn_out, units=RNN_NEURONS[-1] * 4, kernel_initializer=VS_INIT,
                                                kernel_regularizer=None, activation=tf.nn.leaky_relu)

                    dense = tf.layers.dense(dense_mid, units=N_SERIES, kernel_initializer=RND_INIT,
                                            kernel_regularizer=None, activation=None)

                    dense_expand = tf.expand_dims(dense, axis=2)
                    dense_out.append(dense_expand)

        with tf.name_scope("generator_output"):
            output = tf.concat(dense_out, axis=2)

            for p in [Z, conditionals, Z_all, len(rnn_outputs), rnn_outputs, dense_out, output]:
                print(p)

        # Count trainable parameters in graph.
        n_trainable_params = np.sum(
                [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables(scope=GENERATOR_SCOPE)])
        print(f'Generator trainable parameters: {n_trainable_params}')

        # Add optional L2 regularization.
        if L2_REGULARIZATION > 0.:
            l2 = sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables(scope=GENERATOR_SCOPE) if
                     not ("Bias" in tf_var.name))
            regularization_cost = L2_REGULARIZATION * l2
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value=regularization_cost)

    return output, keep_prob


def create_rnn_disc_graph(X, conditionals, reuse=False):
    with tf.variable_scope(DISCRIMINATOR_SCOPE, reuse=reuse):

        with tf.name_scope("discriminator_input"):

            keep_prob = tf.placeholder_with_default(KEEP_PROB, shape=())

            X_all = tf.concat([X, conditionals], axis=1)

            X_sequences = tf.unstack(tf.transpose(X_all, perm=[2, 0, 1]))

        # Make neural network layers.
        with tf.name_scope("discriminator_rnn"):

            RNN_cell = partial(tf.contrib.rnn.GRUCell, activation=tf.nn.tanh, reuse=reuse, kernel_initializer=XAV_INIT)

            cells = [RNN_cell(num_units=neurons) for neurons in RNN_NEURONS]

            cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob) for cell in cells]

            multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)

            rnn_outputs, states = tf.contrib.rnn.static_rnn(multi_layer_cell, X_sequences, dtype=tf.float32)

        with tf.name_scope("discriminator_intermediate_dense"):
            dense_varscope = 'RNN_DISC_TimeDistributed'
            dense_out = list()
            for r, rnn_out in enumerate(rnn_outputs):
                dense_reuse = False if (reuse is False and r == 0) else True
                with tf.variable_scope(dense_varscope, reuse=dense_reuse):
                    dense = tf.layers.dense(rnn_out, units=RNN_NEURONS[-1], kernel_initializer=VS_INIT,
                                            kernel_regularizer=None, activation=tf.nn.leaky_relu)
                    dense_out.append(dense)

        with tf.name_scope("discriminator_output"):
            pre_output_concat = tf.concat(dense_out, axis=1)

            output = tf.layers.dense(pre_output_concat, units=1, activation=None, kernel_initializer=RND_INIT,
                                     kernel_regularizer=None, name='output')

            for p in [X, conditionals, X_all, X_sequences, len(rnn_outputs), rnn_outputs, states, pre_output_concat,
                      output]:
                print(p)

            # Count trainable parameters in graph.
            n_trainable_params = np.sum(
                    [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables(scope=DISCRIMINATOR_SCOPE)])
            print(f'Discriminator trainable parameters: {n_trainable_params}')

            # Add optional L2 regularization.
            if L2_REGULARIZATION > 0.:
                l2 = sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables(scope=DISCRIMINATOR_SCOPE) if
                         not ("Bias" in tf_var.name))
            regularization_cost = L2_REGULARIZATION * l2
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value=regularization_cost)

    return output, keep_prob
