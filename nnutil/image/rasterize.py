#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018, Abdó Roig-Maranges <abdo.roig@gmail.com>
#
# This file is part of 'nnutil'.
#
# This file may be modified and distributed under the terms of the 3-clause BSD
# license. See the LICENSE file for details.

import tensorflow as tf

def raster_indices(coord, shape):
    with tf.name_scope("raster_indices"):
        if type(shape) in set([list, tuple]):
            shape = tf.constant(shape, dtype=coord.dtype)
        else:
            shape = tf.cast(shape, dtype=coord.dtype)

        indexes = coord * (shape - 1)
        indexes = tf.cast(tf.round(indexes), dtype=tf.int32)

    return indexes

def rasterize(coord, values, shape):
    assert(len(coord.shape) == 2)
    assert(len(shape) == coord.shape[-1])

    with tf.name_scope("rasterize"):
        size = tf.shape(coord)[0]

        if values is None:
            values = tf.ones(shape=(size,), dtype=tf.int32)

        value_shape = tf.shape(values)[1:]
        raster_shape = tf.concat([
            tf.constant(shape, dtype=tf.int32),
            value_shape
        ], axis=-1)

        indexes = raster_indices(coord, shape)

        # NOTE: this accumulates values if several points map to the same pixel.
        # TODO: make it take value from a single point
        raster = tf.scatter_nd(
            indexes,
            values,
            raster_shape)

    return raster

def pick(coord, raster):
    dimension = coord.shape[-1]

    with tf.name_scope("pick"):
        indexes = raster_indices(coord, tf.shape(raster)[0:tf.shape(coord)[-1]])
        values = tf.gather_nd(raster, indexes)

    return values
