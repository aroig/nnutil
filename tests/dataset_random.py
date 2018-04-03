import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl


class Dataset_Random(unittest.TestCase):
    def test_dataset_random(self):
        tf.set_random_seed(42)
        ds = nl.dataset.random(shape=(2, 3))

        with tf.Session() as sess:
            it = ds.make_initializable_iterator()
            feature = it.get_next()

            sess.run([it.initializer])
            data = sess.run([feature['image']])

        np.testing.assert_array_equal(data[0].shape, (2, 3))


if __name__ == '__main__':
    unittest.main()
