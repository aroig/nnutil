import inspect

import tensorflow as tf
import numpy as np

from .segment import Segment
from ..math import approximate_identity

class Unet(Segment):
    def __init__(self, encoder, decoder, **kwargs):
        self._encoder = encoder
        self._decoder = decoder
        assert(len(self._encoder) == len(self._decoder))

        super(Unet, self).__init__(layers=self._encoder + self._decoder, **kwargs)

    def build(self, input_shape):
        pass
        # TODO: assert shapes

    def call(self, inputs, **kwargs):
        x = inputs
        encoder_outputs = [x]
        for l in self._encoder:
            x = self._compose(l, x, kwargs)
            encoder_outputs.append(x)

        y = x
        decoder_outputs = [y]
        for l, x in zip(self._decoder, reversed(encoder_outputs[:-1])):
            y = self._compose(l, y, kwargs)
            x = approximate_identity(x, y.shape)

            y = tf.concat([x, y], axis=-1)
            decoder_outputs.append(y)

        output = y

        self._outputs = encoder_outputs[:-1] + decoder_outputs

        return output

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for l in self._layers:
            shape = l.compute_output_shape(shape)
        return shape
