import tensorflow as tf

class Segment(tf.layers.Layer):
    def __init__(self, layers, residual=False, activation=None, activity_regularizer=None,
                 trainable=True, name=None, **kwargs):
        super(Segment, self).__init__(trainable=trainable,
                                      name=name,
                                      activity_regularizer=activity_regularizer,
                                      **kwargs)

        self._residual = residual
        self._activation = activation
        self._layers = layers

    def build(self, input_shape):
        shape = input_shape
        for l in self._layers:
            l.build(shape)
            shape = l.compute_output_shape

        self.built = True

    def call(self, inputs):
        x = inputs
        for l in self._layers:
            x = l.apply(x)

        if self._residual:
            output = inputs + x
        else:
            output = x

        if self._activation is not None:
            output = self._activation(output)

        return output

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for l in self._layers:
            shape = l.compute_output_shape(shape)
        return shape
