import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nn

class Layer_Residual(unittest.TestCase):
    def test_layer_residual(self):
        tf.set_random_seed(42)
        with tf.Session() as sess:
            dense = tf.layers.Dense(units=2)
            layer = nn.layers.Residual(layers=[dense])

            x0 = tf.constant([[1, 2]], dtype=tf.float32)
            x = layer.apply(x0)
            y = dense.apply(x0)

            sess.run([tf.global_variables_initializer()])
            data = sess.run([x, x0 + y])

        self.assertEqual((1, 2), data[0].shape)
        self.assertEqual((1, 2), data[1].shape)
        np.testing.assert_array_almost_equal(data[0], data[1], decimal=5)


if __name__ == '__main__':
    unittest.main()
