import tensorflow as tf

class RangeNormalization(tf.layers.Layer):
    def __init__(self, minval, maxval, axis=None, epsilon=1e-5, activation=None, **kwargs):
        self._minval = minval
        self._maxval = maxval

        self._axis = axis
        self._epsilon = epsilon
        self._activation = activation

        super(RangeNormalization, self).__init__(**kwargs)

    @property
    def axis(self):
        return self._axis

    @property
    def minval(self):
        return self._minval

    @property
    def maxval(self):
        return self._maxval

    def call(self, inputs):
        if self._axis is None:
            self._axis = list(range(0, len(inputs.shape)-1))
        axis = [d+1 for d in self._axis]

        minval = tf.reduce_min(inputs, axis=axis, keepdims=True)
        maxval = tf.reduce_max(inputs, axis=axis, keepdims=True)

        outputs = self._minval + (self._maxval - self._minval) * (inputs - minval) / (maxval - minval + self._epsilon)

        if self._activation is not None:
            outputs = self._activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
