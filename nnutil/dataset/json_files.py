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
        self._path_key = path_key
        self._flatten_lists = flatten_lists
        self._input_spec = input_spec

        files_dataset = tf.data.Dataset.list_files(
            os.path.join(self._directory, self._glob),
            shuffle=shuffle)

        files_dataset = files_dataset.filter(self.path_exists)

        dataset = files_dataset.flat_map(self.load_content)
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

    def path_exists(self, path):
        return tf.py_func(lambda x: os.path.exists(x.decode()), [path], [tf.bool], stateful=False)

    def load_content(self, path):
        content = tf.data.Dataset.from_tensors(tf.read_file(path))
        ds = parse_json(content, self._input_spec, flatten_lists=self._flatten_lists)

        if self._path_key is not None:
            ds_path = tf.data.Dataset.from_tensors({self._path_key: path}).repeat()
            ds = merge([ds, ds_path])

        return ds

    def attach_path(self, feature, path):
        feature[self._path_key] = path
        return feature

def json_files(directory, input_spec, **kwargs):
    return JSONFiles(directory, input_spec, **kwargs)
