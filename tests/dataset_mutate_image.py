import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl


class Dataset_MutateImage(unittest.TestCase):
    def test_dataset_mutate_image_contrast(self):
        tf.set_random_seed(42)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image")
        ds = nl.dataset.image_files(directory=path, glob='*.bmp', shape=(3, 2, 3), shuffle=False)
        ds = nl.dataset.mutate_image(ds, contrast=[0.5, 2], seed=42)

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['image']])

        self.assertEqual((3, 2, 3), data[0].shape)
        for i in range(0, 3):
            np.testing.assert_array_almost_equal(
                data[0][..., i],
                np.array([[1, 0], [0, 1], [1, 1]]),
                decimal=5)

    def test_dataset_mutate_image_contrast(self):
        tf.set_random_seed(42)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image")
        ds = nl.dataset.image_files(directory=path, glob='*.bmp', shape=(3, 2, 3), shuffle=False)
        ds = nl.dataset.mutate_image(ds, brightness=0.5, seed=42)

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['image']])

        self.assertEqual((3, 2, 3), data[0].shape)
        for i in range(0, 3):
            np.testing.assert_array_almost_equal(
                data[0][..., i],
                np.array([[1, 0.45227], [0.45227, 1], [1, 1]]),
                decimal=5)


if __name__ == '__main__':
    unittest.main()
