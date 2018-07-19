import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nn

class Layer_Cyinder(unittest.TestCase):
    def test_layer_cyinder(self):
        tf.set_random_seed(42)
        with tf.Session() as sess:
            layer = nn.layers.Cylinder([
                tf.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')
            ], axis=0, padding=1)

            x0 = tf.constant([[[1], [2], [3]]], dtype=tf.float32)
            y0 = tf.constant([[[2], [2], [2]]], dtype=tf.float32)
            y = layer.apply(x0)

            data = sess.run([y0, y])

        self.assertEqual(data[0].shape, data[1].shape)
        np.testing.assert_array_almost_equal(data[0], data[1], decimal=5)


if __name__ == '__main__':
    unittest.main()
