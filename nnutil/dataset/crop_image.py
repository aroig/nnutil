import tensorflow as tf
import numpy as np
import multiprocessing

class CropImage(tf.data.Dataset):
    def __init__(self, dataset, shape, image_key=None, crop_key=None):
        self._shape = shape
        self._parallel = int (0.8 * multiprocessing.cpu_count())

        if image_key is None:
            image_key = 'image'

        self._image_key = image_key

        if crop_key is None:
            self._crop_window_fn = None

        elif type(crop_key) == str:
            self._crop_window_fn = lambda feature: feature[crop_key]

        elif  hasattr(crop_key, '__call__'):
            self._crop_window_fn = crop_key

        else:
            raise Exception("Cannot handle crop_window")

        if self._crop_window_fn is not None:
            # dataset = dataset.filter(self.is_crop_window_contained)
            dataset = dataset.map(self.crop_image, num_parallel_calls=self._parallel)
        else:
            dataset = dataset.map(self.resize_image, num_parallel_calls=self._parallel)

        self._dataset = dataset

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

    def is_crop_window_contained(self, feature):
        crop_window = tf.cast(self._crop_window_fn(feature), dtype=tf.float32)
        image_shape = tf.cast(tf.shape(feature[self._image_key]), dtype=tf.float32)

        zero = tf.constant(0, dtype=tf.float32)

        C0 = tf.less_equal(zero, crop_window[1])
        C1 = tf.less_equal(zero, crop_window[0])
        C2 = tf.less_equal(crop_window[0] + crop_window[2], image_shape[0])
        C3 = tf.less_equal(crop_window[1] + crop_window[3], image_shape[1])

        return tf.logical_and(tf.logical_and(C0, C1), tf.logical_and(C2, C3))

    def crop_image(self, feature):

        # (y0, x0, y1, x1) crop window in relative coordinates
        crop_window = self._crop_window_fn(feature)

        image = feature[self._image_key]

        # NOTE: We rotate a standardized image, so that the padding is done with a value that becomes zero
        mu, variance = tf.nn.moments(image, axes=[0, 1, 2])
        norm_image = tf.image.per_image_standardization(image)

        norm_image = tf.image.crop_and_resize(
            tf.expand_dims(norm_image, axis=0),
            tf.expand_dims(crop_window, axis=0),
            tf.constant([0], dtype=tf.int32),
            self._shape[0:2],
            extrapolation_value=0)

        image = tf.clip_by_value(mu + tf.sqrt(variance) * norm_image, 0.0, 1.0)

        image = tf.squeeze(image, axis=0)
        image.set_shape(self._shape)
        feature[self._image_key] = image

        return feature

    def resize_image(self, feature):
        image = feature[self._image_key]

        if self._shape is not None:
            # resize_image wants a static shape, but the kernel does
            # not seem to use it. Let's fake it.
            image.set_shape([None, None, None])
            image = tf.image.resize_images(image, size=self._shape[0:2], method=tf.image.ResizeMethod.BILINEAR)
            image.set_shape(self._shape)

        feature[self._image_key] = image
        return feature


def crop_image(dataset, shape, image_key=None, crop_key=None):
    return CropImage(
        dataset,
        shape,
        image_key=image_key,
        crop_key=crop_key)
