import os

import tensorflow as tf
import numpy as np

class Random(tf.data.Dataset):
    def __init__(self, shape, seed=None):
        """shape: (height, width, channels)"""
        self._shape = shape

        dataset = tf.data.Dataset.from_tensors(tf.random_uniform(shape=shape, minval=0, maxval=1, seed=seed))
        dataset = dataset.repeat()
        dataset = dataset.map(self.prepare_feature)
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

    def prepare_feature(self, data):
        feature = {
            'path': tf.constant("test", dtype=tf.string),
            'image': data
        }

        return feature

def random(shape, **kwargs):
    return Random(shape, **kwargs)
