import numpy as np
import tensorflow as tf

from .classification_model import ClassificationModel

class NLConvNetModel(ClassificationModel):
    def __init__(self, name, shape, labels):
        super().__init__(name, shape, labels)

    def model_network(self, x, training):
        self._layers = self.model_layers()

        if self.shape[-1] == 1:
            x = tf.image.rgb_to_grayscale(x)

        x = tf.reshape(x, shape=(-1,) + self.shape)

        for l in self.layers:
            if type(l) == tf.layers.Dropout:
                x = l.apply(x, training=training)
            else:
                x = l.apply(x)

        return x


