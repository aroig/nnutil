import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl


class Dataset_ParseJSON(unittest.TestCase):

    def test_dataset_parse_json_private(self):
        tf.set_random_seed(42)
        raw = '{"a": {"b": 1}, "c": 3}'
        input_spec = {
            "a": {"b": nl.TensorSpec((), tf.int32)},
            "c": nl.TensorSpec((), tf.int32)
        }

        ds = tf.data.Dataset.from_tensor_slices(tf.constant([raw], dtype=tf.string))
        ds = nl.dataset.parse_json(ds, input_spec=input_spec)

        flat = ds.flatten_to_numpy({"a" : { "b": 1 }, "c": 3 })
        np.testing.assert_array_equal(np.array(1), flat[0])
        np.testing.assert_array_equal(np.array(3), flat[1])

        flat = ds.parse_json_fn(raw)
        np.testing.assert_array_equal(np.array(1), flat[0])
        np.testing.assert_array_equal(np.array(3), flat[1])

    def test_dataset_parse_json_parse_private_flatten_lists(self):
        tf.set_random_seed(42)
        raw = '{"a": [{"b": 1}, {"b": 2}], "c": 3}'

        input_spec = {
            "a": {"b": nl.TensorSpec((), tf.int32)},
            "c": nl.TensorSpec((), tf.int32)
        }

        ds = tf.data.Dataset.from_tensor_slices(tf.constant([raw], dtype=tf.string))
        ds = nl.dataset.parse_json(ds, input_spec=input_spec)

        flat = ds.parse_json_flatten_lists_fn(raw)
        np.testing.assert_array_equal(np.array([1, 2]), flat[0])
        np.testing.assert_array_equal(np.array([3, 3]), flat[1])

    def test_dataset_parse_json_1(self):
        tf.set_random_seed(42)
        raw = '{"a": 1, "b": 2}'

        input_spec = {
            "a": nl.TensorSpec((1,), tf.int32),
            "b": nl.TensorSpec((1,), tf.int32)
        }

        ds = tf.data.Dataset.from_tensor_slices(tf.constant([raw], dtype=tf.string))
        ds = nl.dataset.parse_json(ds, input_spec=input_spec)

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['a'], feature['b']])

        np.testing.assert_array_equal(np.array([1]), data[0])
        np.testing.assert_array_equal(np.array([2]), data[1])

    def test_dataset_parse_json_2(self):
        tf.set_random_seed(42)
        raw = '{"a": [[1, 1], [2, 2]], "b": 5}'

        input_spec = {
            "a": nl.TensorSpec((2,2), tf.int32),
            "b": nl.TensorSpec((1,), tf.float32)
        }

        ds = tf.data.Dataset.from_tensor_slices(tf.constant([raw], dtype=tf.string))
        ds = nl.dataset.parse_json(ds, input_spec=input_spec)

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['a'], feature['b']])

        np.testing.assert_array_equal(np.array([[1, 1], [2, 2]]), data[0])
        np.testing.assert_array_equal(np.array([5.0]), data[1])

    def test_dataset_parse_json_3(self):
        tf.set_random_seed(42)

        raw = '{"a": [{ "b": 1 }, { "b": 2 }], "c": 3}'

        input_spec = {
            "a": {"b": nl.TensorSpec((), tf.int32)},
            "c": nl.TensorSpec((), tf.int32)
        }

        ds = tf.data.Dataset.from_tensor_slices(tf.constant([raw], dtype=tf.string))
        ds = nl.dataset.parse_json(ds, input_spec=input_spec, flatten_lists=True)

        with tf.Session() as sess:
            it = ds.make_one_shot_iterator()
            feature = it.get_next()

            data = sess.run([feature['a']['b'], feature['c']])
            np.testing.assert_array_equal(np.array(1), data[0])
            np.testing.assert_array_equal(np.array(3), data[1])

            data = sess.run([feature['a']['b'], feature['c']])
            np.testing.assert_array_equal(np.array(2), data[0])
            np.testing.assert_array_equal(np.array(3), data[1])

if __name__ == '__main__':
    unittest.main()
