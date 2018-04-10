import os

import tensorflow as tf
import numpy as np

from tensorflow.python.data.util import nest

from .parse_json import parse_json

class JSONFiles(tf.data.Dataset):
    def __init__(self, directory, input_spec,
                 glob='*.json', flatten_lists=False, shuffle=True):
        self._directory = directory
        self._glob = glob

        flat_shapes = [spec.shape for spec in nest.flatten(input_spec)]
        input_shapes = nest.pack_sequence_as(input_spec, flat_shapes)

        flat_types = [spec.dtype for spec in nest.flatten(input_spec)]
        input_types = nest.pack_sequence_as(input_spec, flat_types)

        dataset = tf.data.Dataset.list_files(os.path.join(self._directory, self._glob),
                                             shuffle=shuffle)

        dataset = dataset.map(self.load_content)
        dataset = parse_json(dataset, input_shapes, input_types, flatten_lists=flatten_lists)

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

    def load_content(self, path):
        content = tf.read_file(path)
        return content

def json_files(directory, input_spec,
               flatten_lists=False, glob='*.json', shuffle=True):
    return JSONFiles(directory, input_spec, flatten_lists=flatten_lists, glob=glob, shuffle=shuffle)
