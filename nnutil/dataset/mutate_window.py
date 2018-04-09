import tensorflow as tf
import numpy as np

class MutateWindow(tf.data.Dataset):
    def __init__(self, dataset, window_key=None,
                 scale=None, keep_aspect=False, xoffset=None, yoffset=None, seed=None):

        if window_key is None:
            window_key = 'window'
        self._window_key = window_key

        self._seed = seed

        self._scale = self._make_multiplicative_range(scale)
        self._xoffset = self._make_additive_range(xoffset)
        self._yoffset = self._make_additive_range(yoffset)

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

    def _as_variant_tensor(self):
        return self._dataset._as_variant_tensor()

    def do_mutation(self, feature):
        window = tf.cast(feature[self._window_key], dtype=tf.float32)

        height = window[2]
        width = window[3]

        center_y = window[0] + height/2
        center_x = window[1] + width/2

        if self._scale is not None:
            alpha = tf.random_uniform((), minval=self._scale[0], maxval=self._scale[1],
                                      dtype=tf.float32, seed=self._seed)

            height = alpha * height;

            # TODO: maybe add option to keep aspect ratio
            if not self._keep_aspect:
                alpha = tf.random_uniform((), minval=self._scale[0], maxval=self._scale[1],
                                          dtype=tf.float32, seed=self._seed)

            width = alpha * width;

        if self._xoffset is not None:
            delta = tf.random_uniform((), minval=self._xoffset[0], maxval=self._xoffset[1],
                                      dtype=tf.float32, seed=self._seed)
            center_x = center_x + delta * width

        if self._yoffset is not None:
            delta = tf.random_uniform((), minval=self._yoffset[0], maxval=self._yoffset[1],
                                      dtype=tf.float32, seed=self._seed)
            center_y = center_y + delta * height

        yoffset = tf.cast(tf.round(center_y - height/2), dtype=tf.int32)
        xoffset = tf.cast(tf.round(center_x - width/2), dtype=tf.int32)

        height = tf.cast(tf.round(height), dtype=tf.int32)
        width = tf.cast(tf.round(width), dtype=tf.int32)

        feature[self._window_key] = tf.stack([yoffset, xoffset, height, width])

        return feature


def mutate_window(dataset, window_key=None, scale=None, xoffset=None, yoffset=None, seed=None):
    return MutateWindow(dataset,
                        window_key=window_key,
                        scale=scale,
                        xoffset=xoffset,
                        yoffset=yoffset,
                        seed=seed)
