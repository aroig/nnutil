import tensorflow as tf

class Reshape(tf.layers.Layer):
    def __init__(self, shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self._shape = shape

    def call(self, inputs):
        outputs = tf.reshape(inputs, shape = (tf.shape(inputs)[0],) + self._shape)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self._shape
