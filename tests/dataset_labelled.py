import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl


class Dataset_Labelled(unittest.TestCase):
    def test_dataset_labelled(self):
        tf.set_random_seed(42)

        ds1 = tf.data.Dataset.from_tensor_slices({"value": tf.constant([1, 2, 3], dtype=tf.int32)})

        ds2 = tf.data.Dataset.from_tensor_slices({"value": tf.constant([10, 20, 30], dtype=tf.int32)})

        ds = nl.dataset.labelled({ "a": ds1, "b": ds2 }, label_str_key="label")

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['value'], feature['label']])
            self.assertEqual(10, data[0])
            self.assertEqual("b", data[1].decode())


if __name__ == '__main__':
    unittest.main()
