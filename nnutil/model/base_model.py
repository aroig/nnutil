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

    def gradient_norm(self, f, variables):
        grad = [tf.gradient(f, v) for v in variables]
        return sum([tf.reduce_sum(tf.square(g)) for g in grad])
