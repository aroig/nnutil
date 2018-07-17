import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl


class Util_SliceAxis(unittest.TestCase):
    def test_util_slice_axis_1(self):
        tf.set_random_seed(42)

        with tf.Session() as sess:
            x = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=tf.int32)
            y = nl.util.slice_axis(x, [0, 2], axis=0)
            res = sess.run(y)

        np.testing.assert_array_almost_equal(
            res,
            np.array([[0, 1, 2], [3, 4, 5]]),
            decimal=5)

    def test_util_slice_axis_1(self):
        tf.set_random_seed(42)

        with tf.Session() as sess:
            x = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=tf.int32)
            y = nl.util.slice_axis(x, [-2, 0], axis=0)
            res = sess.run(y)

        np.testing.assert_array_almost_equal(
            res,
            np.array([[3, 4, 5], [6, 7, 8]]),
            decimal=5)
