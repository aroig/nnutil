import os

import tensorflow as tf
import numpy as np

class TextFiles(tf.data.Dataset):
    def __init__(self, directory, glob, shuffle=True):
        self._directory = os.path.realpath(directory)
        self._glob = glob

        dataset = tf.data.Dataset.list_files(os.path.join(self._directory, self._glob),
                                             shuffle=shuffle)
        dataset = dataset.map(self.load_content)

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

    def _inputs(self):
        return []

    def _as_variant_tensor(self):
        return self._dataset._as_variant_tensor()

    def load_content(self, txt_path):
        content = tf.read_file(txt_path)

        feature = {
            'path': txt_path,
            'content': content
        }

        return feature

def text_files(directory, **kwargs):
    return TextFiles(directory, **kwargs)
