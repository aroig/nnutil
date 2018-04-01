import numpy as np
import tensorflow as tf

class BaseModel:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def compose_layers(self, layers, x, training):
        for l in layers:
            if type(l) == tf.layers.Dropout:
                x = l.apply(x, training=training)
            else:
                x = l.apply(x)
        return x

    def layer_gradients(self, f, layers=[]):
        return [tf.gradient(f, v) for l in layers for v in l.variables]
