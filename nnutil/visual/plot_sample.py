import os

import tensorflow as tf
import numpy as np

from .mosaic import MosaicWindow


def plot_sample(dataset, image_fn=None, path_fn=None, label_fn=None, box_fn=None):
    with tf.Session() as sess:
        win = MosaicWindow(sess, dataset,
                           image_fn=image_fn, path_fn=path_fn,
                           label_fn=label_fn, box_fn=box_fn)
