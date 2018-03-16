import unittest
import os

import numpy as np
import tensorflow as tf
import nlnnutil as nl


class Dataset_Interleave(unittest.TestCase):
    def setUp(self):
        self._data = [ 0, 10, 20, 30, 1, 11, 21, 31 ]
        self._datasets = [tf.data.Dataset.from_tensor_slices(tf.constant([10*i, 10*i+1], dtype=tf.int32)) for i in range(0, 4)]

    def test_dataset_interleave(self):
        dataset = nl.dataset.interleave(self._datasets)
        dataset = dataset.batch(len(self._data))

        with tf.Session() as sess:
            it = dataset.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature])
            data = data[0]

        np.testing.assert_array_equal(data, np.array(self._data))

    def test_dataset_interleave_zipped(self):
        dataset = nl.dataset.interleave([tf.data.Dataset.zip((self._datasets[0], self._datasets[1])),
                                         tf.data.Dataset.zip((self._datasets[2], self._datasets[3]))])
        dataset = dataset.batch(int(len(self._data) / 2))

        with tf.Session() as sess:
            it = dataset.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature])
            data = data[0]

        np.testing.assert_array_equal(data, np.array([[0, 20, 1, 21], [10, 30, 11, 31]]))


if __name__ == '__main__':
    unittest.main()
