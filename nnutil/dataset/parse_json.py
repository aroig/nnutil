import os
import json
import itertools

import tensorflow as tf
import numpy as np

from tensorflow.python.data.util import nest


class ParseJSON(tf.data.Dataset):
    def __init__(self, dataset, input_shapes, input_types, flatten_lists=False):
        self._input_shapes = input_shapes
        self._input_types = input_types
        self._flatten_lists = flatten_lists

        self._flat_types = nest.flatten(input_types)
        self._flat_shapes = nest.flatten_up_to(input_types, input_shapes)

        dataset = dataset.flat_map(self.parse_json)

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

    def filter_keys(self, data, input_types):
        """Only keep keys in the nested structure data, that match an entry in input_types
        """
        if type(data) == tuple and type(input_types) == tuple:
            return tuple([self.filter_keys(x, t) for x, t in zip(data, input_types)])

        elif type(data) == dict and type(input_types) == dict:
            return {k: self.filter_keys(data[k], t) for k, t in input_types.items()}

        elif type(data) == list and type(input_types) == dict:
            return [self.filter_keys(x, input_types) for x in data]

        elif type(data) == list and type(input_types) == tuple:
            return [self.filter_keys(x, input_types) for x in data]

        else:
            return data

    def flatten_lists(self, data, input_types):
        """Remove lists in inner nodes of a nested structure, by instantiating multiple
           nested structures replacing the lists with every possible combination of its elements
        """

        if type(data) == tuple and type(input_types) == tuple:
            flat_inner = [self.flatten_lists(x, t) for x, t in zip(data, input_types)]
            return list(itertools.product(*flat_inner))

        elif type(data) == dict and type(input_types) == dict:
            flat_inner = [[(k, xi) for xi in self.flatten_lists(data[k], t)]
                          for k, t in input_types.items()]
            return [dict(tup) for tup in itertools.product(*flat_inner)]

        elif type(data) == list and type(input_types) == dict:
            flat_inner = [self.flatten_lists(x, input_types) for x in data]
            return list(itertools.chain(*flat_inner))

        elif type(data) == list and type(input_types) == tuple:
            flat_inner = [self.flatten_lists(x, input_types) for x in data]
            return list(itertools.chain(*flat_inner))

        else:
            return [data]

    def flatten_to_numpy(self, data):
        flat_data = nest.flatten_up_to(self._input_types, data)
        return [np.reshape(np.array(x, dtype=t.as_numpy_dtype), s)
                for x, s, t in zip(flat_data, self._flat_shapes, self._flat_types)]

    def parse_json_fn(self, raw):
        data = json.loads(raw)
        data = self.filter_keys(data, self._input_types)
        return self.flatten_to_numpy(data)

    def parse_json_flatten_lists_fn(self, raw):
        data = json.loads(raw)
        data = self.filter_keys(data, self._input_types)

        data_list = self.flatten_lists(data, self._input_types)
        flat_data = [self.flatten_to_numpy(data) for data in data_list]

        return [np.stack(xtuple) for xtuple in zip(*flat_data)]

    def parse_json(self, raw):
        if self._flatten_lists:
            flat_feature = tf.py_func(self.parse_json_flatten_lists_fn,
                                      [raw], self._flat_types, stateful=False)

            feature = nest.pack_sequence_as(self._input_types, flat_feature)

            return tf.data.Dataset.from_tensor_slices(feature)

        else:
            flat_feature = tf.py_func(self.parse_json_fn,
                                      [raw], self._flat_types, stateful=False)

            feature = nest.pack_sequence_as(self._input_types, flat_feature)

            return tf.data.Dataset.from_tensors(feature)


def parse_json(dataset, input_shapes, input_types, flatten_lists=False):
    return ParseJSON(dataset, input_shapes, input_types, flatten_lists=flatten_lists)

