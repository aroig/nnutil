import os

import tensorflow as tf
import numpy as np

from tensorflow.python.data.util import nest

from .parse_json import parse_json
from .merge import merge

class JSONFiles(tf.data.Dataset):
    def __init__(self, directory, input_spec,
                 glob='*.json', flatten_lists=False, path_key=None, shuffle=True):
        self._directory = os.path.realpath(directory)
        self._glob = glob

        files_dataset = tf.data.Dataset.list_files(
            os.path.join(self._directory, self._glob),
            shuffle=shuffle)

        dataset = files_dataset.map(self.load_content)
        dataset = parse_json(dataset, input_spec, flatten_lists=flatten_lists)

        self._path_key = path_key

        if self._path_key is not None:
            dataset = tf.data.Dataset.zip((dataset, files_dataset)).map(self.attach_path)

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

    def attach_path(self, feature, path):
        feature[self._path_key] = path
        return feature

def json_files(directory, input_spec,
               flatten_lists=False, path_key=None, glob='*.json', shuffle=True):
    return JSONFiles(directory, input_spec, flatten_lists=flatten_lists, path_key=path_key, glob=glob, shuffle=shuffle)
