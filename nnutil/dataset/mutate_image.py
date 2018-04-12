import tensorflow as tf
import numpy as np

class MutateImage(tf.data.Dataset):
    def __init__(self, dataset, image_key=None,
                 hue=None, brightness=None, contrast=None, saturation=None,
                 hflip=False, noise=None, seed=None):

        if image_key is None:
            image_key = 'image'
        self._image_key = image_key

        self._seed = seed

        self._brightness = brightness
        self._hue = hue
        self._contrast = self._make_multiplicative_range(contrast)
        self._saturation = self._make_multiplicative_range(saturation)

        self._hflip = hflip
        self._noise = noise

        self._dataset = dataset.map(self.do_mutation)

    def _make_multiplicative_range(self, x):
        if x is None:
            return None

        if type(x) == float and x >= 1.:
            return (1/x, x)

        elif type(x) == tuple or type(x) == list:
            return (x[0], x[1])

        else:
            raise Exception("Unknown multiplicative range")

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
        image = feature[self._image_key]

        if self._hue is not None and image.shape[-1] == 3:
            image = tf.image.random_hue(image,
                                        max_delta=self._hue,
                                        seed=self._seed)

        if self._brightness is not None:
            image = tf.image.random_brightness(image,
                                               max_delta=self._brightness,
                                               seed=self._seed)

        if self._contrast is not None:
            image = tf.image.random_contrast(image,
                                             lower=self._contrast[0],
                                             upper=self._contrast[1],
                                             seed=self._seed)

        if self._saturation is not None and image.shape[-1] == 3:
            image = tf.image.random_saturation(image,
                                               lower=self._saturation[0],
                                               upper=self._saturation[1],
                                               seed=self._seed)

        if self._hflip:
            image = tf.image.random_flip_left_right(image)

        if self._noise is not None:
            image = image + tf.random_normal(image.shape, mean=0, stddev=self._noise)

        feature[self._image_key] = tf.clip_by_value(image, 0.0, 1.0)

        return feature


def mutate_image(dataset, image_key=None,
                 hue=None, brightness=None, contrast=None, saturation=None,
                 hflip=False, noise=None, seed=None):
    return MutateImage(dataset,
                       image_key=image_key,
                       hue=hue,
                       brightness=brightness,
                       contrast=contrast,
                       saturation=saturation,
                       hflip=hflip,
                       noise=noise,
                       seed=seed)
