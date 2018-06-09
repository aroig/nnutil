#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# nnutil - Neural network utilities for tensorflow
# Copyright (c) 2018, Abdó Roig-Maranges <abdo.roig@gmail.com>
#
# This file is part of 'nnutil'.
#
# This file may be modified and distributed under the terms of the 3-clause BSD
# license. See the LICENSE file for details.

import tensorflow as tf

def mosaic(image_matrix, name=None, border=None):
    """ image_matrix: (row, col, height, width, channels)"""
    if name is None:
        name = "Mosaic"

    with tf.name_scope(name):
        shape = tf.shape(image_matrix)
        mosaic_shape = shape[0:2]
        image_shape = shape[2:5]

        image_array = tf.reshape(image_matrix, shape=(-1,) + image_shape)
        mosaic = tf.transpose(image_array, perm=[0, 2, 1, 3, 4])

        shape = tf.shape(mosaic)
        mosaic = tf.reshape(mosaic, shape=(1, shape[0] * shape[1], shape[2] * shape[3], shape[4]))

    return mosaic

def batch_mosaic(image, aspect_ratio=None, name=None):
    if aspect_ratio is None:
        aspect_ratio = 1.0

    if name is None:
        name = "BatchMosaic"

    with tf.name_scope(name):
        size = tf.shape(image)[0]
        alpha = tf.sqrt(tf.cast(size, dtype=tf.float32) / aspect_ratio)

        ncols = tf.cast(tf.ceil(aspect_ratio * alpha), dtype=tf.int32)
        nrows = tf.cast(tf.floor(alpha), dtype=tf.int32)
        nrows = tf.cond(nrows * ncols <= size, lambda: nrows, lambda: nrows+1)

        # add padding to image stack
        rank = tf.rank(image)
        padding_shape = tf.shape(image) + tf.one_hot(0, rank, dtype=tf.int32)*(nrows*ncols - 2*size)
        padded_img = tf.concat([image, tf.zeros(shape=padding_shape, dtype=tf.float32)], axis=0)

        # create mosaic matrix
        shape = tuple(image.shape)[1:]
        image_matrix = tf.reshape(padded_img, shape=(nrows, ncols) + shape)

        return mosaic(image_matrix)


def confusion_mosaic(image, nlabels, labels, predictions, name=None):
    if name is None:
        name = "ConfusionMosaic"

    with tf.name_scope(name):
        size = tf.shape(labels)[0]

        image_ex = tf.concat([tf.zeros((1,) + tuple(image.shape[1:]), dtype=image.dtype), image], axis=0)

        indexes = tf.stack([
            tf.cast(labels, dtype=tf.int32),
            tf.cast(predictions, dtype=tf.int32),
            tf.range(0, size, dtype=tf.int32)
        ], axis=-1)

        unique_indexes = tf.scatter_nd(indexes, tf.range(1, size+1), shape=(nlabels, nlabels, size))
        unique_indexes = tf.reduce_max(unique_indexes, axis=-1)

        image_matrix = tf.gather_nd(image_ex, tf.expand_dims(unique_indexes, axis=-1))

        return mosaic(image_matrix)
