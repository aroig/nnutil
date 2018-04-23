import tensorflow as tf
from .segment import Segment

class Bottleneck2D(Segment):
    def __init__(self, filters=None, kernel_size=None, strides=None, padding="same",
                 data_format="channels_last", depth_multiplier=1, activation=None, **kwargs):

        super(Bottleneck2D, self).__init__(layers=[
            tf.layers.SeparableConv2D(filters=depth_multiplier * filters,
                                      kernel_size=1,
                                      strides=1,
                                      data_format=data_format,
                                      activation=activation),

            tf.layers.Conv2D(filters=depth_multiplier * filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             data_format=data_format,
                             activation=activation),

            tf.layers.SeparableConv2D(filters=filters,
                                      kernel_size=1,
                                      strides=1,
                                      data_format=data_format,
                                      activation=None)
        ], **kwargs)

        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._data_format = data_format
        self._depth_multiplier = depth_multiplier
        self._activation = activation
