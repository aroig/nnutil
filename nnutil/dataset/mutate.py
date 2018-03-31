import tensorflow as tf
import numpy as np

class Mutate(tf.data.Dataset):
    def __init__(self, dataset, colorspace=True):
        """shape: (height, width, channels)"""
        self._colorspace = colorspace

        self._dataset = dataset.map(self.do_mutation)

    @property
    def output_classes(self):
        return self._dataset.output_classes

    @property
    def output_shapes(self):
        return self._dataset.output_shapes

    @property
    def output_types(self):
        return self._dataset.output_types

    def _as_variant_tensor(self):
        return self._dataset._as_variant_tensor()

    def do_mutation(self, feature):
        image = feature['image']

        if self._colorspace:
            image = tf.image.random_brightness(image, max_delta=0.3)
            image = tf.image.random_contrast(image, lower=0.6, upper=1.8)

            image = tf.cond(tf.equal(tf.shape(image)[-1], 3),
                            lambda: tf.image.random_saturation(image, lower=0.6, upper=1.8),
                            lambda: image)

        feature['image'] = tf.clip_by_value(image, 0.0, 1.0)
        return feature


def mutate(dataset, colorspace=True):
    return Mutate(dataset, colorspace=colorspace)
