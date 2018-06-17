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


def to_rgb(image):
    shape = tf.shape(image)
    depth = shape[-1]

    output_shape = tf.concat([shape[:-1], [3]], axis=0)
    image_array = tf.reshape(image, shape=(-1, shape[-3], shape[-2], shape[-1]))

    image_rgb = tf.cond(
        depth >= 3,
        lambda: image_array[..., 0:3],
        lambda: tf.image.grayscale_to_rgb(image_array[..., 0:1]))

    output = tf.reshape(image_rgb, shape=output_shape)
    return output


def to_grayscale(image):
    shape = tf.shape(image)
    depth = shape[-1]

    output_shape = tf.concat([shape[:-1], [1]], axis=0)
    image_array = tf.reshape(image, shape=(-1, shape[-3], shape[-2], shape[-1]))

    image_grey = tf.cond(
        depth >= 3,
        lambda: tf.image.rgb_to_grayscale(image_array[..., 0:3]),
        lambda: image_array[..., 0:1])

    output = tf.reshape(image_grey, shape=output_shape)
    return output
