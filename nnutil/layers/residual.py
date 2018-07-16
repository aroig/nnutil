import inspect

import tensorflow as tf
import numpy as np

from ..math import approximate_identity
from .segment import Segment

class Residual(Segment):
    def __init__(self, layers, activation=None, **kwargs):
        self._residual_activation = activation
        super(Residual, self).__init__(layers, activation=None, **kwargs)

    def build(self, input_shape):
        output_shape = self.compute_output_shape(input_shape)
        assert(input_shape.ndims == output_shape.ndims)

        super(Residual, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = super(Residual, self).call(inputs, **kwargs)

        shape = x.shape.as_list()
        output = approximate_identity(inputs, shape) + x

        if self._residual_activation is not None:
            output = self._residual_activation(output)

        return output
