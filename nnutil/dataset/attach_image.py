import tensorflow as tf
import numpy as np

class AttachImage(tf.data.Dataset):
    def __init__(self, dataset, shape, image_key=None, image_path=None, crop_window=None):
        self._shape = shape

        if image_key is None:
            image_key = 'image'

        self._image_key = image_key

        if image_path is None:
            self._image_path_fn = lambda feature: feature["image"]

        elif type(image_path) == str:
            self._image_path_fn = lambda feature: feature[image_path]

        elif hasattr(image_path, '__call__'):
            self._image_path_fn = image_path

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

        dataset = dataset.map(self.read_image)

        if self._crop_window_fn is not None:
            dataset = dataset.filter(self.is_crop_window_contained)
            dataset = dataset.map(self.crop_image)

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
        C2 = tf.less(crop_window[0] + crop_window[2], image_shape[0])
        C3 = tf.less(crop_window[1] + crop_window[3], image_shape[1])

        return tf.logical_and(tf.logical_and(C0, C1), tf.logical_and(C2, C3))

    def read_image(self, feature):
        path = self._image_path_fn(feature)
        data = tf.read_file(path)
        nchannels = 0 if self._shape is None else self._shape[2]

        image = tf.image.decode_image(data, channels=nchannels)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        feature[self._image_key] = image
        return feature

    def crop_image(self, feature):
        crop_window = tf.cast(self._crop_window_fn(feature), dtype=tf.int32)
        image = feature[self._image_key]

        image = tf.image.crop_to_bounding_box(image,
                                              crop_window[0],
                                              crop_window[1],
                                              crop_window[2],
                                              crop_window[3])

        if self._shape is not None:
            # resize_image wants a static shape, but the kernel does
            # not seem to use it. Let's fake it.
            image.set_shape([None, None, None])
            image = tf.image.resize_images(image, size=self._shape[0:2])
            image.set_shape(self._shape)

        feature[self._image_key] = image

        return feature


def attach_image(dataset, shape, image_key=None, image_path=None, crop_window=None):
    return AttachImage(dataset, shape,
                       image_key=image_key,
                       image_path=image_path,
                       crop_window=crop_window)
