import os

import tensorflow as tf
import numpy as np

from .attach_image import attach_image

class ImageFiles(tf.data.Dataset):
    def __init__(self, directory, shape, glob='*', shuffle=True, **kwargs):
        """shape: (height, width, channels)"""
        self._directory = os.path.realpath(directory)
        self._glob = glob
        self._shape = shape

        dataset = tf.data.Dataset.list_files(
            os.path.join(self._directory, self._glob),
            shuffle=shuffle)

        dataset = dataset.map(lambda x: {'path': x})

        dataset = attach_image(
            dataset,
            self._shape,
            image_path='path',
            **kwargs)

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

    def _inputs(self):
        return []

    def _as_variant_tensor(self):
        return self._dataset._as_variant_tensor()

def image_files(directory, shape, **kwargs):
    return ImageFiles(directory, shape, **kwargs)
