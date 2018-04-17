import tensorflow as tf

class LayerNormalization(tf.layers.Layer):
    def __init__(self, axis=None, epsilon=1e-5, **kwargs):
        self._axis = axis
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        axis = self._axis
        if axis is None:
            axis = list(range(1, len(inputs.shape)))
        else:
            axis = [d+1 for d in axis]

        mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
        outputs = (inputs - mean) / tf.sqrt(variance + self._epsilon)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
