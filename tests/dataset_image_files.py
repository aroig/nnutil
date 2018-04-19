import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl


class Dataset_ImageFiles(unittest.TestCase):
    def test_dataset_image_files_bmp_1(self):
        tf.set_random_seed(42)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./image")
        ds = nl.dataset.image_files(directory=path, glob='2x3.bmp', shape=(3, 2, 3), shuffle=False)

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

    def test_dataset_image_files_bmp_2(self):
        tf.set_random_seed(42)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./image")
        ds = nl.dataset.image_files(directory=path, glob='2x3.bmp', shape=(3, 2, 1), shuffle=False)

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['image']])

        self.assertEqual((3, 2, 1), data[0].shape)
        np.testing.assert_array_almost_equal(
            data[0][..., 0],
            np.array([[1, 0], [0, 1], [1, 1]]),
            decimal=5)

    def test_dataset_image_files_bmp_3(self):
        tf.set_random_seed(42)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./image")
        ds = nl.dataset.image_files(directory=path, glob='2x3-red.bmp', shape=(3, 2, 3), shuffle=False)

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['image']])

        self.assertEqual((3, 2, 3), data[0].shape)

        np.testing.assert_array_almost_equal(
            data[0][..., 0],
            np.array([[1, 1], [1, 1], [1, 1]]),
            decimal=5)

        for i in range(1, 3):
            np.testing.assert_array_almost_equal(
                data[0][..., i],
                np.array([[1, 0], [0, 1], [1, 1]]),
                decimal=5)

    def test_dataset_image_files_jpg(self):
        tf.set_random_seed(42)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./image")
        ds = nl.dataset.image_files(directory=path, glob='2x3.jpg', shape=(3, 2, 3), shuffle=False)

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['image']])

        self.assertEqual((3, 2, 3), data[0].shape)
        for i in range(0, 3):
            np.testing.assert_array_almost_equal(
                data[0][..., i],
                np.array([[1, 0], [0, 1], [1, 1]]),
                decimal=1)


if __name__ == '__main__':
    unittest.main()
