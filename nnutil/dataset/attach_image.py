import tensorflow as tf
import numpy as np
import multiprocessing

from .crop_image import crop_image

class AttachImage(tf.data.Dataset):
    def __init__(self, dataset, shape, image_key=None, shape_key=None, image_path=None, image_src=None, crop_window=None):
        self._shape = shape
        self._parallel = int (0.8 * multiprocessing.cpu_count())

        if image_key is None:
            image_key = 'image'

        self._image_key = image_key
        self._shape_key = shape_key

        if type(image_src) == str:
            self._image_data_fn = lambda feature: feature[image_src]

        elif hasattr(image_src, '__call__'):
            self._image_data_fn = lambda feature: image_src(feature)

        elif type(image_path) == str:
            self._image_data_fn = lambda feature: tf.read_file(feature[image_path])

        elif hasattr(image_path, '__call__'):
            self._image_data_fn = lambda feature: tf.read_file(image_path(feature))

        else:
            raise Exception("Cannot handle image_path")

        if crop_window is None:
            self._crop_window_fn = None

        elif type(crop_window) == str:
            self._crop_window_fn = lambda feature: feature[crop_window]

        elif  hasattr(image_path, '__call__'):
            self._crop_window_fn = crop_window

        else:
            raise Exception("Cannot handle crop_window")

        dataset = dataset.map(self.read_image, num_parallel_calls=self._parallel)

        if self._crop_window_fn is not None:
            dataset = crop_image(
                dataset,
                self._shape,
                image_key=self._image_key,
                crop_key=self._crop_window_fn)

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

    def read_image(self, feature):
        data = self._image_data_fn(feature)

        image = tf.image.decode_image(data)
        if self._shape is not None:
            if self._shape[2] == 1:
                image = tf.cond(
                    tf.equal(tf.shape(image)[2], 1),
                    lambda: image,
                    lambda: tf.image.rgb_to_grayscale(image))

            elif self._shape[2] == 3:
                image = tf.cond(
                    tf.equal(tf.shape(image)[2], 3),
                    lambda: image,
                    lambda: tf.image.grayscale_to_rgb(image))

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        feature[self._image_key] = image

        if self._shape_key is not None:
            feature[self._shape_key] = tf.shape(image)

        return feature


def attach_image(dataset, shape, image_key=None, shape_key=None, image_path=None, image_src=None, crop_window=None):
    return AttachImage(
        dataset,
        shape,
        image_key=image_key,
        shape_key=shape_key,
        image_path=image_path,
        image_src=image_src,
        crop_window=crop_window)
