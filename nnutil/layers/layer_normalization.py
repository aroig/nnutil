import tensorflow as tf

class LayerNormalization(tf.layers.Layer):
    def __init__(self, axis=None, epsilon=1e-5, activation=None, **kwargs):
        self._axis = axis
        self._epsilon = epsilon
        self._activation = activation

        super(LayerNormalization, self).__init__(**kwargs)

    @property
    def axis(self):
        return self._axis

    def call(self, inputs):
        if self._axis is None:
            self._axis = list(range(0, len(inputs.shape)-1))
        axis = [d+1 for d in self._axis]

        mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)

        # Use the bessel correction for the variance, to match NLConvNet normalization
        shape = tf.shape(inputs)
        N = tf.cast(tf.reduce_prod([shape[d] for d in axis]), dtype=tf.float32)
        variance = variance * N / (N-1)

        outputs = (inputs - mean) / tf.sqrt(variance + self._epsilon)

        if self._activation is not None:
            outputs = self._activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
