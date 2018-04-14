import os

import tensorflow as tf
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow.python.data.util import nest

from .mosaic import MosaicWindow


def plot_sample(dataset):
    with tf.Session() as sess:
        win = MosaicWindow(sess, dataset)


def print_sample(dataset):
    with tf.Session() as sess:
        it = dataset.make_one_shot_iterator()
        feature = it.get_next()
        while True:
            flat = nest.flatten(feature)
            flat_val = sess.run(flat)
            np_feature = nest.pack_sequence_as(feature, flat_val)
            print(np_feature)
