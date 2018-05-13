
import tensorflow as tf
import numpy as np

class AttachRaw(tf.data.Dataset):
    def __init__(self, dataset, content_key=None, file_path=None):
        self._parallel = 4

        if content_key is None:
            content_key = 'content'

        self._content_key = content_key

        if file_path is None:
            self._file_path_fn = lambda feature: feature["image"]

        elif type(file_path) == str:
            self._file_path_fn = lambda feature: feature[file_path]

        elif hasattr(file_path, '__call__'):
            self._file_path_fn = file_path

        else:
            raise Exception("Cannot handle file_path")

        dataset = dataset.map(self.read_file, num_parallel_calls=self._parallel)

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

    def read_file(self, feature):
        path = self._file_path_fn(feature)
        data = tf.read_file(path)

        feature[self._content_key] = data
        return feature

def attach_raw(dataset, content_key=None, file_path=None):
    return AttachRaw(
        dataset,
        content_key=content_key,
        file_path=file_path)
