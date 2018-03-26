import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl


class Layer_Reshape(unittest.TestCase):
    def test_layer_reshape(self):
        with tf.Session() as sess:
            layer = nl.layer.Reshape(shape=(2, 2))
            x = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
            x = layer.apply(x)
            data = sess.run([x])

        self.assertEqual((1, 2, 2), data[0].shape)
        np.testing.assert_array_almost_equal(data[0], np.array([[[1, 2], [3, 4]]]), decimal=5)


if __name__ == '__main__':
    unittest.main()
