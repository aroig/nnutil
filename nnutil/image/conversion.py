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
    image_shape = shape[-3:]

    image_array = tf.reshape(image, shape=(-1, image_shape[0], image_shape[1], image_shape[2]))

    image_rgb = tf.cond(
        tf.equal(image_shape[-1], 3),
        lambda: image_array,
        lambda: tf.image.grayscale_to_rgb(image_array))

    output = tf.reshape(image_rgb, shape=shape)
    return output


def to_grayscale(image):
    shape = tf.shape(image)
    image_shape = shape[-3:]

    image_array = tf.reshape(image, shape=(-1, image_shape[0], image_shape[1], image_shape[2]))

    image_grey = tf.cond(
        tf.equal(image_shape[-1], 1),
        lambda: image_array,
        lambda: tf.image.rgb_to_grayscale(image_array))

    output = tf.reshape(image_grey, shape=shape)
    return output
