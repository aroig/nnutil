import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl


class Dataset_AttachImage(unittest.TestCase):
    def test_dataset_mutate_window_scale(self):
        tf.set_random_seed(42)
        ds = tf.data.Dataset.from_tensors({
            'crop': tf.constant([0, 0, 20, 20], dtype=tf.int32)
        })
        ds = nl.dataset.mutate_window(ds, window_key='crop', scale=[0.5, 2], seed=42)

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['crop']])

            np.testing.assert_array_equal(data[0], np.array([-9, -9, 39, 39]))

    def test_dataset_mutate_window_xoffset(self):
        tf.set_random_seed(42)
        ds = tf.data.Dataset.from_tensors({
            'crop': tf.constant([0, 0, 20, 20], dtype=tf.int32)
        })
        ds = nl.dataset.mutate_window(ds, window_key='crop', xoffset=[-0.5, 0.5], seed=42)

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['crop']])

            np.testing.assert_array_equal(data[0], np.array([0, 9, 20, 20]))

    def test_dataset_mutate_window_yoffset(self):
        tf.set_random_seed(42)
        ds = tf.data.Dataset.from_tensors({
            'crop': tf.constant([0, 0, 20, 20], dtype=tf.int32)
        })
        ds = nl.dataset.mutate_window(ds, window_key='crop', yoffset=[-0.5, 0.5], seed=42)

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['crop']])

            np.testing.assert_array_equal(data[0], np.array([9, 0, 20, 20]))


if __name__ == '__main__':
    unittest.main()
