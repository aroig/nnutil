import tensorflow as tf
import numpy as np
import math
import multiprocessing

class MutateImage(tf.data.Dataset):
    def __init__(self, dataset, image_key=None,
                 hue=None, brightness=None, contrast=None, saturation=None,
                 hflip=False, rotate=None,
                 gaussian_noise=None, impulse_noise=None, seed=None):

        if image_key is None:
            image_key = 'image'
        self._image_key = image_key

        self._seed = seed
        self._parallel = int (0.5 * multiprocessing.cpu_count())

        self._brightness = brightness
        self._hue = hue
        self._contrast = self._make_multiplicative_range(contrast)
        self._saturation = self._make_multiplicative_range(saturation)

        self._hflip = hflip
        self._rotate = self._make_additive_range(rotate)

        self._gaussian_noise = self._make_additive_range(gaussian_noise)
        self._impulse_noise = self._make_additive_range(impulse_noise)

        self._dataset = dataset.map(self.do_mutation, num_parallel_calls=self._parallel)

    def _make_multiplicative_range(self, x):
        if x is None:
            return None

        if type(x) == float and x >= 1.:
            return (1/x, x)

        elif type(x) == tuple or type(x) == list:
            return (x[0], x[1])

        else:
            raise Exception("Unknown multiplicative range")

    def _make_additive_range(self, x):
        if x is None:
            return None

        if type(x) == float and x >= 0:
            return (-x, x)

        elif type(x) == tuple or type(x) == list:
            return (x[0], x[1])

        else:
            raise Exception("Unknown additive range")

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

        if self._rotate is not None:
            angle = (2 * math.pi / 360) * tf.random_uniform((), minval=self._rotate[0], maxval=self._rotate[1], dtype=tf.float32)

            # NOTE: We rotate a standardized image, so that the padding is done with a value that becomes zero
            mu, variance = tf.nn.moments(image, axes=[-3, -2, -1])
            epsilon = 1e-4
            sigma = tf.sqrt(variance) + epsilon
            norm_image = (image - mu) / sigma

            norm_image = tf.contrib.image.rotate(norm_image, angle, interpolation='BILINEAR')
            image = tf.clip_by_value(mu + sigma * norm_image, 0.0, 1.0)

        if self._gaussian_noise is not None:
            sigma = tf.random_uniform(
                (),
                minval=self._gaussian_noise[0],
                maxval=self._gaussian_noise[1],
                dtype=tf.float32)
            noise = tf.random_normal(image.shape, mean=0, stddev=sigma)
            image = image + noise

        if self._impulse_noise is not None:
            prob = tf.random_uniform(
                (),
                minval=self._impulse_noise[0],
                maxval=self._impulse_noise[1],
                dtype=tf.float32)
            dist = tf.distributions.Categorical(
                probs=[0.5*prob, 1 - prob, 0.5*prob],
                dtype=tf.float32)
            noise = dist.sample(image.shape) - 1
            image = image + noise

        feature[self._image_key] = tf.clip_by_value(image, 0.0, 1.0)

        return feature


def mutate_image(dataset, **kwargs):
    return MutateImage(dataset, **kwargs)
