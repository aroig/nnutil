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

    def variable_summaries(self, gradients):
        # Gradients and weight summaries
        for grad, weight in gradients:
            # Compute per-batch normalized norms
            axis = [i for i in range(1, len(grad.shape))]
            grad_norm = tf.sqrt(tf.reduce_mean(tf.square(grad), axis=axis))
            weight_norm = tf.sqrt(tf.reduce_mean(tf.square(weight), axis=axis))

            name = weight.name.replace(':', '_')
            tf.summary.histogram('{}/weight'.format(name), weight_norm)
            tf.summary.histogram('{}/grad'.format(name), grad_norm)
