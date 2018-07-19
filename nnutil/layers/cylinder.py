import math

import tensorflow as tf

from tensorflow.python.framework import tensor_shape

from .segment import Segment
from ..util import slice_axis

class Cylinder(Segment):
    def __init__(self, layers, axis=None, padding=None, **kwargs):
        super(Cylinder, self).__init__(layers, **kwargs)
        if axis is None:
            axis = 0
        self._axis = axis

        if padding is None:
            padding = 3
            first_layer = self.layers[0]
            if hasattr(first_layer, "kernel_size") and \
               self._axis < len(first_layer.kernel_size):
                padding = int(math.ceil(first_layer.kernel_size[self._axis] / 2.0))

        self._padding = padding

    def build(self, input_shape):
        super(Cylinder, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return super(Cylinder, self).compute_output_shape(input_shape)

    def call(self, inputs):
        size = self._padding
        begin_slice = slice_axis(inputs, [0, size], axis=self._axis + 1)
        end_slice = slice_axis(inputs, [-size, 0], axis=self._axis + 1)

        inputs_ex = tf.concat([
            end_slice,
            inputs,
            begin_slice
        ], axis=self._axis + 1)

        outputs = super(Cylinder, self).call(inputs_ex)
        outputs = slice_axis(outputs, [size, -size], axis=self._axis + 1)

        return outputs
