import os
import pprint

import tensorflow as tf
import numpy as np

from tensorflow.python.data.util import nest


def print_sample(dataset):
    pp = pprint.PrettyPrinter(indent=4)

    with tf.Session() as sess:
        it = dataset.make_one_shot_iterator()
        feature = it.get_next()
        try:
            while True:
                flat = nest.flatten(feature)
                flat_val = sess.run(flat)
                np_feature = nest.pack_sequence_as(feature, flat_val)
                pp.pprint(np_feature)

        except tf.errors.OutOfRangeError:
            pass
