
import tensorflow as tf

def slice_axis(x, rg, axis=0):
    # TODO: normalize axis
    rank = len(x.shape)

    assert(0 <= axis)
    assert(axis < rank)

    size = x.shape[axis]
    if size is None:
        raise Exception("Slicing dimension must be statically known")

    assert(rg[0] < size and -size <= rg[0])
    assert(rg[1] < size and -size <= rg[1])
    assert(rg[0] < rg[1])

    if rg[0] < 0:
        rg[0] = size - rg[0]

    if rg[1] < 0:
        rg[1] = size - rg[1]

    begin = [0] * rank
    begin[axis] = rg[0]
    begin = tf.constant(begin, dtype=tf.int32)

    shape = tf.unstack(tf.shape(x))
    shape[axis] = tf.constant(rg[1] - rg[0], dtype=tf.int32)
    shape = tf.stack(shape)

    return tf.slice(x, begin, shape)
