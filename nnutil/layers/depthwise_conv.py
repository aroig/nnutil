
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.layers.convolutional import _Conv
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils

class _DepthwiseConv(_Conv):
  def __init__(self,
               rank,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               depth_multiplier=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(_DepthwiseConv, self).__init__(
        rank=rank,
        filters=None,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        **kwargs)

    self.depth_multiplier = depth_multiplier

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')

    input_dim = input_shape[channel_axis].value
    self.input_spec = base.InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})

    kernel_shape = self.kernel_size + (input_dim, self.depth_multiplier)

    self.kernel = self.add_variable(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)

    if self.use_bias:
      self.bias = self.add_variable(name='bias',
                                    shape=(input_dim * self.depth_multiplier,),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    trainable=True,
                                    dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    raise NotImplementedError


class DepthwiseConv1D(_DepthwiseConv):
  def __init__(self,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               depth_multiplier=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(SeparableConv1D, self).__init__(
        rank=1,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        depth_multiplier=depth_multiplier,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        **kwargs)

  def call(self, inputs):
    if self.data_format == 'channels_last':
      strides = (1,) + self.strides * 2 + (1,)
      spatial_start_dim = 1
    else:
      strides = (1, 1) + self.strides * 2
      spatial_start_dim = 2

    # Explicitly broadcast inputs and kernels to 4D.
    # TODO(fchollet): refactor when a native separable_conv1d op is available.
    inputs = array_ops.expand_dims(inputs, spatial_start_dim)
    kernel = array_ops.expand_dims(self.kernel, 0)
    dilation_rate = (1,) + self.dilation_rate

    outputs = tf.nn.depthwise_conv2d(
        inputs,
        kernel,
        strides=strides,
        padding=self.padding.upper(),
        dilations=dilation_rate,
        data_format=utils.convert_data_format(self.data_format, ndim=4))

    if self.use_bias:
      outputs = tf.nn.bias_add(
          outputs,
          self.bias,
          data_format=utils.convert_data_format(self.data_format, ndim=4))

    outputs = array_ops.squeeze(outputs, [spatial_start_dim])

    if self.activation is not None:
      return self.activation(outputs)

    return outputs


class DepthwiseConv2D(_DepthwiseConv):
  def __init__(self,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               depth_multiplier=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(DepthwiseConv2D, self).__init__(
        rank=2,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        depth_multiplier=depth_multiplier,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        **kwargs)

  def call(self, inputs):
    # Apply the actual ops.
    if self.data_format == 'channels_last':
      strides = (1,) + self.strides + (1,)
    else:
      strides = (1, 1) + self.strides
    outputs = tf.nn.depthwise_conv2d(
        inputs,
        self.kernel,
        strides=strides,
        padding=self.padding.upper(),
        rate=self.dilation_rate,
        data_format=utils.convert_data_format(self.data_format, ndim=4))

    if self.use_bias:
      outputs = tf.nn.bias_add(
          outputs,
          self.bias,
          data_format=utils.convert_data_format(self.data_format, ndim=4))

    if self.activation is not None:
      return self.activation(outputs)

    return outputs
