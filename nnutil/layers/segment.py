import inspect

import tensorflow as tf
import numpy as np

class Segment(tf.layers.Layer):
    def __init__(self, layers, activation=None, **kwargs):
        super(Segment, self).__init__(**kwargs)

        self._activation = activation
        self._layers = layers
        self._outputs = None

    def _compose(self, l, x, kwargs):
        sig = [p.name for p in inspect.signature(l.call).parameters.values()]
        args = {k: kwargs[k] for k in set(sig) & set(kwargs.keys())}
        y = l.apply(x, **args)
        return y

    @property
    def layers(self):
        def append_layers(L, l):
            if isinstance(l, Segment):
                for l2 in l.layers:
                    append_layers(L, l2)
            else:
                L.append(l)

        layer_list = []
        for l in self._layers:
            append_layers(layer_list, l)

        return layer_list

    @property
    def depth(self):
        return len(self.layers)

    @property
    def variables(self):
        return [v for l in self._layers for v in l.variables]

    @property
    def trainable_variables(self):
        return [v for l in self._layers for v in l.trainable_variables]

    @property
    def non_trainable_variables(self):
        return [v for l in self._layers for v in l.non_trainable_variables]

    @property
    def losses(self):
        return [loss for l in self._layers for loss in l.losses]

    @property
    def updates(self):
        return [u for l in self._layers for u in l.updates]

    @property
    def size(self):
        return np.sum([np.prod(v.shape) for v in self.variables])

    @property
    def input(self):
        return self._layers[0].input

    @property
    def input_shape(self):
        return self._layers[0].input_shape

    @property
    def output(self):
        return self._layers[-1].output

    @property
    def output_shape(self):
        return self._layers[-1].output_shape

    @property
    def layer_activations(self):
        return self._outputs

    def build(self, input_shape):
        super(Segment, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        self._outputs = [x]

        for l in self._layers:
            x = self._compose(l, x, kwargs)
            self._outputs.append(x)

        output = x

        if self._activation is not None:
            output = self._activation(output)

        return output

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for l in self._layers:
            shape = l.compute_output_shape(shape)
        return shape
