import os

import tensorflow as tf
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow.python.data.util import nest

from .mosaic import MosaicWindow


def plot_sample(dataset, image_fn=None, path_fn=None, label_fn=None, box_fn=None):
    with tf.Session() as sess:
        win = MosaicWindow(sess, dataset,
                           image_fn=image_fn, path_fn=path_fn,
                           label_fn=label_fn, box_fn=box_fn)


def print_sample(dataset):
    with tf.Session() as sess:
        it = dataset.make_one_shot_iterator()
        feature = it.get_next()
        while True:
            flat = nest.flatten(feature)
            flat_val = sess.run(flat)
            np_feature = nest.pack_sequence_as(feature, flat_val)
            print(np_feature)
