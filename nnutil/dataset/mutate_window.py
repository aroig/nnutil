import tensorflow as tf
import numpy as np

class MutateWindow(tf.data.Dataset):
    def __init__(self, dataset, window_key=None, angle_key=None,
                 scale=None, keep_aspect=False, xoffset=None, yoffset=None, seed=None, rotate=None):
        self._input_datasets = [dataset]

        if window_key is None:
            window_key = 'window'
        self._window_key = window_key

        if angle_key is None:
            angle_key = 'angle'
        self._angle_key = angle_key

        self._seed = seed

        self._scale = self._make_multiplicative_range(scale)
        self._xoffset = self._make_additive_range(xoffset)
        self._yoffset = self._make_additive_range(yoffset)
        self._rotate = self._make_additive_range(rotate)

        self._keep_aspect = keep_aspect

        self._dataset = dataset.map(self.do_mutation)

    def _make_additive_range(self, x):
        if x is None:
            return None

        if type(x) == float and x >= 0:
            return (-x, x)

        elif type(x) == tuple or type(x) == list:
            return (x[0], x[1])

        else:
            raise Exception("Unknown additive range")

    def _make_multiplicative_range(self, x):
        if x is None:
            return None

        if type(x) == float and x >= 1.:
            return (1/x, x)

        elif type(x) == tuple or type(x) == list:
            return (x[0], x[1])

        else:
            raise Exception("Unknown multiplicative range")

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
        return list(self._input_datasets)

    def _as_variant_tensor(self):
        return self._dataset._as_variant_tensor()

    def do_mutation(self, feature):
        window = tf.cast(feature[self._window_key], dtype=tf.float32)

        p0 = window[0:2]
        p1 = window[2:4]

        height = tf.abs(p1[0] - p0[0])
        width = tf.abs(p1[1] - p0[1])

        center = (p0 + p1) / 2

        v0 = p0 - center
        v1 = p1 - center

        if self._scale is not None:
            alpha0 = tf.random_uniform((), minval=self._scale[0], maxval=self._scale[1],
                                       dtype=tf.float32, seed=self._seed)
            if self._keep_aspect:
                alpha = tf.stack([alpha0, alpha0])

            else:
                alpha1 = tf.random_uniform((), minval=self._scale[0], maxval=self._scale[1],
                                           dtype=tf.float32, seed=self._seed)
                alpha = tf.stack([alpha0, alpha1])

            v0 = v0 * alpha
            v1 = v1 * alpha

        if self._xoffset is not None:
            delta1 = tf.random_uniform((), minval=self._xoffset[0], maxval=self._xoffset[1],
                                       dtype=tf.float32, seed=self._seed)
        else:
            delta1 = 0

        if self._yoffset is not None:
            delta0 = tf.random_uniform((), minval=self._yoffset[0], maxval=self._yoffset[1],
                                      dtype=tf.float32, seed=self._seed)
        else:
            delta0 = 0

        center = center + tf.stack([delta0 * height, delta1 * width])

        mutated_window = tf.concat([center + v0, center + v1], axis=0)

        feature[self._window_key] = mutated_window

        if self._rotate is not None:
            angle = feature.get(self._angle_key, tf.constant(0, dtype=tf.float32))
            delta = tf.random_uniform((), minval=self._rotate[0], maxval=self._rotate[1], dtype=tf.float32, seed=self._seed)
            feature[self._angle_key] = angle + delta

        return feature


def mutate_window(dataset, **kwargs):
    return MutateWindow(dataset, **kwargs)
