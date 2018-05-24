import tensorflow as tf
from .segment import Segment
from .depthwise_conv import DepthwiseConv2D

class Bottleneck2D(Segment):
    def __init__(self, filters=None, kernel_size=None, strides=None, padding="same",
                 data_format="channels_last", depth_multiplier=1, activation=None,
                 kernel_regularizer=None, residual=False):

        super(Bottleneck2D, self).__init__(layers=[
            tf.layers.Conv2D(filters=depth_multiplier * filters,
                             kernel_size=1,
                             strides=1,
                             data_format=data_format,
                             kernel_regularizer=kernel_regularizer,
                             activation=activation),

            DepthwiseConv2D(kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            depth_multiplier=1,
                            data_format=data_format,
                            kernel_regularizer=kernel_regularizer,
                            activation=activation),

            tf.layers.Conv2D(filters=filters,
                             kernel_size=1,
                             strides=1,
                             data_format=data_format,
                             kernel_regularizer=kernel_regularizer,
                             activation=None)
        ], residual=residual, activation=None)

        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._data_format = data_format
        self._depth_multiplier = depth_multiplier
