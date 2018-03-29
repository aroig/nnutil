import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nn


class Layer_Segment(unittest.TestCase):
    def test_layer_segment(self):
        with tf.Session() as sess:
            dense = tf.layers.Dense(units=2)
            layer = nn.layers.Segment(layers=[dense])

            x0 = tf.constant([[1, 2]], dtype=tf.float32)
            x = layer.apply(x0)
            y = dense.apply(x0)

            sess.run([tf.global_variables_initializer()])
            data = sess.run([x, y])

        self.assertEqual((1, 2), data[0].shape)
        self.assertEqual((1, 2), data[1].shape)
        np.testing.assert_array_almost_equal(data[0], data[1], decimal=5)

    def test_layer_segment_residual(self):
        with tf.Session() as sess:
            dense = tf.layers.Dense(units=2)
            layer = nn.layers.Segment(layers=[dense], residual=True)

            x0 = tf.constant([[1, 2]], dtype=tf.float32)
            x = layer.apply(x0)
            y = dense.apply(x0)

            sess.run([tf.global_variables_initializer()])
            data = sess.run([x, x0 + y])

        self.assertEqual((1, 2), data[0].shape)
        self.assertEqual((1, 2), data[1].shape)
        np.testing.assert_array_almost_equal(data[0], data[1], decimal=5)

    def test_layer_segment_dropout(self):
        with tf.Session() as sess:
            dropout = tf.layers.Dropout()
            layer = nn.layers.Segment(layers=[dropout])

            x0 = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
            x = layer.apply(x0, training=False)

            data = sess.run([x0, x])

        self.assertEqual((1, 4), data[0].shape)
        self.assertEqual((1, 4), data[1].shape)
        np.testing.assert_array_almost_equal(data[0], data[1], decimal=5)




if __name__ == '__main__':
    unittest.main()
