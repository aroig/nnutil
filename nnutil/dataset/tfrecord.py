import os

import tensorflow as tf
import numpy as np

from tensorflow.python.framework.tensor_spec import TensorSpec

class TFRecord(tf.data.Dataset):
    def __init__(self, path, input_spec):
        self._path = path
        self._input_spec = input_spec

        dataset = tf.data.TFRecordDataset([self._path])
        dataset = dataset.map(self.parse_example)

        self._dataset = dataset

    @property
    def output_classes(self):
        return self._dataset.output_classes

    @property
    def output_shapes(self):
        return self._dataset.output_shapes

    @property
    def output_types(self):
        return self._dataset.output_types

    def _as_variant_tensor(self):
        return self._dataset._as_variant_tensor()

    def parse_spec(self, input_spec):
        if type(input_spec) == dict:
            return {k: self.parse_spec(v) for k, v in input_spec.items()}

        elif type(input_spec) == TensorSpec:
            return tf.FixedLenFeature(input_spec.shape, input_spec.dtype)

        elif type(input_spec) == tf.FixedLenFeature:
            return input_spec

        elif type(input_spec) == tf.VarLenFeature:
            return input_spec

        else:
            raise Exception("Unhandled input spec")


    def parse_example(self, example_proto):
        parse_spec = self.parse_spec(self._input_spec)
        parsed_features = tf.parse_single_example(example_proto, parse_spec)

        return parsed_features


def tfrecord(path, input_spec):
    return TFRecord(path=path, input_spec=input_spec)
