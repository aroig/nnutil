import inspect
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

    @property
    def layers(self):
        return self._layers

    @property
    def variables(self):
        return [v for l in self._layers for v in l.variables]

    def build(self, input_shape):
        shape = input_shape
        for l in self._layers:
            l.build(shape)
            shape = l.compute_output_shape(shape)

        super(Segment, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        for l in self._layers:
            sig = [p.name for p in inspect.signature(l.call).parameters.values()]

            args = {k: kwargs[k] for k in set(sig) & set(kwargs.keys())}
            x = l.apply(x, **args)

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
