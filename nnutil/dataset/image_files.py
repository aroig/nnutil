import os

import tensorflow as tf
import numpy as np

class ImageFiles(tf.data.Dataset):
    def __init__(self, directory, glob='*', shape=None):
        """shape: (height, width, channels)"""
        self._directory = directory
        self._shape = shape
        self._glob = glob

        dataset = tf.data.Dataset.list_files(os.path.join(self._directory, self._glob))
        dataset = dataset.map(self.load_image)

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

    def load_image(self, img_path):
        if self._shape is not None:
            nchannels = self._shape[2]
        else:
            nchannels = 0

        data = tf.read_file(img_path)

        # Note: This does not set a shape, as gifs produce a different rank (animated gifs).
        image = tf.image.decode_image(data, channels=nchannels)

        # The specific image decoders, do set a shape
        # image = tf.image.decode_bmp(data, channels=nchannels)
        # image = tf.image.decode_jpeg(data, channels=nchannels)

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        if self._shape is not None:
            # resize_image wants a shape, but the kernel does not use it. Let's fake it.
            image.set_shape([None, None, None])
            image = tf.image.resize_images(image, size=self._shape[0:2])
            image.set_shape(self._shape)

        feature = {
            'path': img_path,
            'image': image,
            'shape': tf.shape(image)
        }

        return feature

def image_files(directory, glob='*', shape=None):
    return ImageFiles(directory, glob=glob, shape=shape)
