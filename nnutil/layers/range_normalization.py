import tensorflow as tf

class RangeNormalization(tf.layers.Layer):
    def __init__(self, minval, maxval, axis=None, epsilon=1e-5, **kwargs):
        self._minval = minval
        self._maxval = maxval

        self._axis = axis
        self._epsilon = epsilon
        super(RangeNormalization, self).__init__(**kwargs)

    @property
    def minval(self):
        return self._minval

    @property
    def maxval(self):
        return self._maxval

    def call(self, inputs):
        axis = self._axis
        if axis is None:
            axis = list(range(1, len(inputs.shape)))
        else:
            axis = [d+1 for d in axis]

        minval = tf.reduce_min(inputs, axis=axis, keepdims=True)
        maxval = tf.reduce_max(inputs, axis=axis, keepdims=True)

        outputs = minval + (inputs - minval) / (maxval - minval + self._epsilon)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
