import os

import tensorflow as tf
import numpy as np

class Merge(tf.data.Dataset):
    def __init__(self, datasets):

        dataset = tf.data.Dataset.zip(tuple(datasets))
        dataset = dataset.map(self.merge_dicts)

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

    def merge_dicts(self, *features):
        res = {}
        for f in features:
            for k in f:
                res[k] = f[k]
        return res


def merge(datasets, **kwargs):
    return Merge(datasets, **kwargs)
