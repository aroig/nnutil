import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nn

class Image_Rasterize(unittest.TestCase):
    def test_image_rasterize_1(self):
        tf.set_random_seed(42)
        with tf.Session() as sess:
            coord = tf.constant([[0.1, 0.1], [0.8, 0.1]], dtype=tf.float32)
            raster = nn.image.rasterize(coord, None, (2, 2))

            data = sess.run(raster)

        self.assertEqual((2, 2), data.shape)
        np.testing.assert_array_almost_equal(
            data,
            np.array([[1, 0], [1, 0]]),
            decimal=4)

    def test_image_rasterize_2(self):
        tf.set_random_seed(42)
        with tf.Session() as sess:
            coord = tf.constant([[0.1, 0.1], [0.8, 0.1]], dtype=tf.float32)
            value = tf.constant([0.5, 0.3], dtype=tf.float32)
            raster = nn.image.rasterize(coord, value, (2, 2))

            data = sess.run(raster)

        self.assertEqual((2, 2), data.shape)
        np.testing.assert_array_almost_equal(
            data,
            np.array([[0.5, 0], [0.3, 0]]),
            decimal=4)

    def test_image_rasterize_3(self):
        tf.set_random_seed(42)
        with tf.Session() as sess:
            coord = tf.constant([[0.1, 0.1], [0.8, 0.1]], dtype=tf.float32)
            value = tf.constant([[0.5, 0.6], [0.2, 0.3]], dtype=tf.float32)
            raster = nn.image.rasterize(coord, value, (2, 2))

            data = sess.run(raster)

        self.assertEqual((2, 2, 2), data.shape)
        np.testing.assert_array_almost_equal(
            data,
            np.array([[[0.5, 0.6], [0, 0]], [[0.2, 0.3], [0, 0]]]),
            decimal=4)

    def test_image_pick_1(self):
        tf.set_random_seed(42)
        with tf.Session() as sess:
            coord = tf.constant([[0.1, 0.1], [0.8, 0.1], [0.9, 0.9]], dtype=tf.float32)
            raster = tf.constant([[1, 0], [1, 0]], dtype=tf.float32)
            value = nn.image.pick(coord, raster)

            data = sess.run(value)

        self.assertEqual((3,), data.shape)
        np.testing.assert_array_almost_equal(
            data,
            np.array([1, 1, 0]),
            decimal=4)

    def test_image_pick_2(self):
        tf.set_random_seed(42)
        with tf.Session() as sess:
            coord = tf.constant([[0.1, 0.1], [0.8, 0.1], [0.9, 0.9]], dtype=tf.float32)
            raster = tf.constant([[[1], [0]], [[1], [0]]], dtype=tf.float32)
            value = nn.image.pick(coord, raster)

            data = sess.run(value)

        self.assertEqual((3, 1), data.shape)
        np.testing.assert_array_almost_equal(
            data,
            np.array([[1], [1], [0]]),
            decimal=4)
