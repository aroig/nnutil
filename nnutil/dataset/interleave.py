import os
import math

import tensorflow as tf
import numpy as np


class Interleave(tf.data.Dataset):
    def __init__(self, datasets):
        """Interleaves the given input datasets.
        """

        self._input_datasets = tuple([ds.repeat() for ds in datasets])
        self._size = len(self._input_datasets)

        dataset = tf.data.Dataset.zip(self._input_datasets)
        dataset = dataset.flat_map(self.serialize_zip)
        self._dataset = dataset

    def serialize_zip(self, *args):
        return tf.data.Dataset.from_tensor_slices(self.recursive_stack(args))

    def recursive_stack(self, args):
        if type(args[0]) == tuple:
            n = len(args[0])
            return tuple([self.recursive_stack([t[i] for t in args]) for i in range(0, n)])

        elif type(args[0]) == dict:
            keys = args[0].keys()
            return {k: self.recursive_stack([t[k] for t in args]) for k in keys}

        else:
            return tf.stack(args)

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


def interleave(datasets):
    return Interleave(datasets)
