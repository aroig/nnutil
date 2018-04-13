import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl

class Dataset_Partition(unittest.TestCase):
    def setUp(self):
        self._data = [ 'a', 'b', 'c', 'd', 'e', 'f' ]
        self._dataA = [x.encode() for x in [ 'a', 'b', 'd', 'e', 'a', 'b' ]]
        self._dataB = [x.encode() for x in [ 'c', 'f', 'c', 'f', 'c', 'f' ]]

    def test_dataset_partition(self):
        tf.set_random_seed(42)
        ds = tf.data.Dataset.from_tensor_slices({'xxx': tf.constant(self._data, dtype=tf.string)})
        ds = tf.data.Dataset.repeat(ds)

        dsA, dsB = nl.dataset.partition(ds, [2, 1], split_field='xxx')
        dsA = tf.data.Dataset.batch(dsA, 6)
        dsB = tf.data.Dataset.batch(dsB, 6)

        with tf.Session() as sess:
            itA = dsA.make_one_shot_iterator()
            itB = dsB.make_one_shot_iterator()

            featureA = itA.get_next()
            featureB = itB.get_next()

            dataA, dataB = sess.run([featureA['xxx'], featureB['xxx']])

        np.testing.assert_array_equal(dataA, np.array(self._dataA))
        np.testing.assert_array_equal(dataB, np.array(self._dataB))


if __name__ == '__main__':
    unittest.main()
