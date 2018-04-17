import tensorflow as tf

class LayerNormalization(tf.layers.Layer):
    def __init__(self, axis, epsilon=1e-5, **kwargs):
        self._axis = axis
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs,
                                       [d+1 for d in self._axis],
                                       keep_dims=True)
        outputs = (inputs - mean) / tf.sqrt(variance + self._epsilon)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
