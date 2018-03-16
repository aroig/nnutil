import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl


class Dataset_Merge(unittest.TestCase):
    def test_dataset_merge(self):
        ds1 = tf.data.Dataset.from_tensors({'a': tf.constant(1, dtype=tf.int32)})

        ds2 = tf.data.Dataset.from_tensors({
            'b': tf.constant(2, dtype=tf.int32),
            'c': tf.constant(3, dtype=tf.int32)
        })

        ds = nl.dataset.merge([ds1, ds2])

        with tf.Session() as sess:
            it1 = ds1.make_one_shot_iterator()

            feature = it1.get_next()
            self.assertEqual(set(feature.keys()), set(['a']))

            data = sess.run([feature['a']])
            self.assertEqual(1, data[0])

            it = ds.make_one_shot_iterator()
            feature = it.get_next()
            self.assertEqual(set(feature.keys()), set(['a', 'b', 'c']))

            data = sess.run([feature['a'], feature['b'], feature['c']])
            self.assertEqual(1, data[0])
            self.assertEqual(2, data[1])
            self.assertEqual(3, data[2])


if __name__ == '__main__':
    unittest.main()
