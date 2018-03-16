import unittest
import os

import numpy as np
import tensorflow as tf
import nlnnutil as nl


class Dataset_ImageFiles(unittest.TestCase):
    def test_dataset_image_files_bmp(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Tests/image")
        ds = nl.dataset.image_files(directory=path, glob='*.bmp', shape=(3, 2, 3))

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['image']])

        self.assertEqual((3, 2, 3), data[0].shape)
        for i in range(0, 3):
            np.testing.assert_array_almost_equal(data[0][...,i], np.array([[1, 0], [0, 1], [1, 1]]), decimal=5)

    def test_dataset_image_files_jpg(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Tests/image")
        ds = nl.dataset.image_files(directory=path, glob='*.jpg', shape=(3, 2, 3))

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['image']])

        self.assertEqual((3, 2, 3), data[0].shape)
        for i in range(0, 3):
            np.testing.assert_array_almost_equal(data[0][...,i], np.array([[1, 0], [0, 1], [1, 1]]), decimal=1)


if __name__ == '__main__':
    unittest.main()
