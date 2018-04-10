import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl

from tensorflow.python.framework.tensor_spec import TensorSpec

class Dataset_JSONFiles(unittest.TestCase):
    def test_dataset_json_files(self):
        tf.set_random_seed(43)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json")

        input_spec = {
            "a": {"b": nl.TensorSpec((2), tf.int32)},
            "c": nl.TensorSpec((), tf.int32)
        }

        ds = nl.dataset.json_files(path, input_spec, shuffle=False)

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature])

        self.assertEqual((2,), data[0]['a']['b'].shape)
        np.testing.assert_array_almost_equal(
                data[0]['a']['b'],
                np.array([1, 2]),
                decimal=5)
        self.assertEqual(data[0]['c'], 3)


if __name__ == '__main__':
    unittest.main()
