import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl


class Dataset_AttachImage(unittest.TestCase):
    def test_dataset_attach_image(self):
        tf.set_random_seed(42)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Tests/image")
        ds = tf.data.Dataset.from_tensors({
            'path': tf.constant(os.path.join(path, "2x3.bmp"), dtype=tf.string)
        })
        ds = nl.dataset.attach_image(ds, shape=(3, 2, 3), image_path='path')

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

    def test_dataset_attach_image_crop(self):
        tf.set_random_seed(42)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Tests/image")
        ds = tf.data.Dataset.from_tensors({
            'path': tf.constant(os.path.join(path, "2x3.bmp"), dtype=tf.string),
            'crop': tf.constant([0, 0, 0.5, 1], dtype=tf.float32)
        })
        ds = nl.dataset.attach_image(ds, shape=(2, 2, 3), image_path='path', crop_window='crop')

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['image']])

        self.assertEqual((2, 2, 3), data[0].shape)
        for i in range(0, 3):
            np.testing.assert_array_almost_equal(
                data[0][..., i],
                np.array([[1, 0], [0, 1]]),
                decimal=5)



if __name__ == '__main__':
    unittest.main()
