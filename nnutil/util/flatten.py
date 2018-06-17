import tensorflow as tf
import numpy as np

def flatten(tensor):
    shape = tensor.shape
    if all([n is not None for n in shape[1:]]):
        dim = np.prod(shape[1:])
        return tf.reshape(tensor, shape=(-1, dim))
    else:
        dim = tf.reduce_prod(tf.shape(tensor)[1:])
        return tf.reshape(tensor, shape=(-1, dim))
