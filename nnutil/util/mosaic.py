import tensorflow as tf
import numpy as np


def mosaic(image):
    # rows and columns
    aspect = 1.0
    size = tf.shape(image)[0]
    alpha = tf.sqrt(tf.cast(size, dtype=tf.float32) / aspect)

    ncols = tf.cast(tf.ceil(aspect * alpha), dtype=tf.int32)
    nrows = tf.cast(tf.floor(alpha), dtype=tf.int32)
    nrows = tf.cond(nrows * ncols <= size, lambda: nrows, lambda: nrows+1)

    # add padding to image stack
    rank = tf.rank(image)
    padding_shape = tf.shape(image) + tf.one_hot(0, rank, dtype=tf.int32) * (nrows * ncols - 2 * size)
    padded_img = tf.concat([image, tf.zeros(shape=padding_shape, dtype=tf.float32)], axis=0)

    # create mosaic
    shape = tuple(image.shape)[1:]
    mosaic = tf.reshape(padded_img, shape=(nrows, ncols) + shape)

    # reshape it
    mosaic = tf.transpose(mosaic, perm=[0, 2, 1, 3, 4])
    shape = tf.shape(mosaic)
    mosaic = tf.reshape(mosaic, shape=(1, shape[0] * shape[1], shape[2] * shape[3], shape[4]))
    return mosaic


def confusion_mosaic(image, nlabels, labels, predictions):
    indexes = tf.stack([
        tf.cast(labels, dtype=tf.int32),
        tf.cast(predictions, dtype=tf.int32)
    ], axis=-1)
    mosaic = tf.scatter_nd(indexes, image, shape=(nlabels, nlabels) + tuple(image.shape)[1:])

    # reshape it
    mosaic = tf.transpose(mosaic, perm=[0, 2, 1, 3, 4])
    shape = tf.shape(mosaic)
    mosaic = tf.reshape(mosaic, shape=(1, shape[0] * shape[1], shape[2] * shape[3], shape[4]))
    return mosaic
