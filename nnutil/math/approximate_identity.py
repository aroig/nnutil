
import tensorflow as tf
import numpy as np

def closest_fraction(x, N):
    """Compute (a, b) such that a/b is closest to x from below, and with 0 <= a, b < N"""
    best_num = 0
    best_den = 1
    best_approx = 0.0

    assert(x >= 0)

    for den in range(1, N):
        num = round(den * x)
        approx = num / den
        if x - approx >= 0 and x - approx < x - best_approx:
            best_num = num
            best_den = den
            best_approx = approx

            # If we are very close, no need to search for more.
            if x - approx < 1e-5:
                break

    return (best_num, best_den)

def _approximate_identity_1d(x, input_shape, output_shape):
    # x: (batch, x, channel)
    assert(len(input_shape) == 2)
    assert(len(output_shape) == 2)

    if (input_shape[0] != output_shape[0]):
        x = tf.reshape(x, shape=(-1, 1, input_shape[0], input_shape[1]))
        x = tf.image.resize_bilinear(x, (1, output_shape[0]))
        x = tf.reshape(x, shape=(-1, output_shape[0], input_shape[1]))

    if input_shape[-1] < output_shape[-1]:
        n = output_shape[-1] - input_shape[-1]
        padding = tf.constant([[0, 0], [0, 0], [0, n]], dtype=tf.int32)
        x = tf.pad(x, padding)

    elif input_shape[-1] > output_shape[-1]:
        x = tf.expand_dims(x, axis=-1)
        x = approximate_identity(x, (-1,) + output_shape + (1,))
        x = tf.squeeze(x, axis=-1)

    return x

def _approximate_identity_2d(x, input_shape, output_shape):
    # x: (batch, x, y, channel)
    assert(len(input_shape) == 3)
    assert(len(output_shape) == 3)

    if (input_shape[0] != output_shape[0] or input_shape[1] != output_shape[1]):
        x = tf.image.resize_bilinear(x, (output_shape[0], output_shape[1]))

    if input_shape[-1] < output_shape[-1]:
        n = int(output_shape[-1] - input_shape[-1])
        padding = tf.constant([[0, 0], [0, 0], [0, 0], [0, n]], dtype=tf.int32)
        x = tf.pad(x, padding)

    elif input_shape[-1] > output_shape[-1]:
        x = tf.expand_dims(x, axis=-1)
        x = approximate_identity(x, (-1,) + output_shape + (1,))
        x = tf.squeeze(x, axis=-1)

    return x

def _approximate_identity_nd(x, input_shape, output_shape):
    # x: (batch, spatial..., channel)
    assert(len(input_shape) > 3)
    assert(len(output_shape) > 3)

    assert(len(input_shape) == len(output_shape))
    rank = len(input_shape)

    if input_shape[-1] < output_shape[-1]:
        x = _approximate_identity_nd(x, input_shape, output_shape[:-1] + (input_shape[-1],))
        n = int(output_shape[-1] - input_shape[-1])
        padding = tf.constant(rank * [[0, 0]] + [[0, n]], dtype=tf.int32)
        x = tf.pad(x, padding)

    elif input_shape[-1] == output_shape[-1]:
        if (input_shape[-2] == output_shape[-2]):
            rest = np.prod(input_shape[-2:])
            x = tf.reshape(x, shape=(-1,) + input_shape[:-2] + (rest,))
            x = approximate_identity(x, (-1,) + output_shape[:-2] + (rest,))
            x = tf.reshape(x, shape=(-1,) + output_shape)

        else:
            x = tf.reshape(x, shape=(-1,) + input_shape[-3:])
            x = approximate_identity(x, (-1,) + output_shape[-3:])
            x = tf.reshape(x, shape=(-1,) + input_shape[:-3] + output_shape[-3:])
            x = _approximate_identity_nd(x, input_shape[:-3] + output_shape[-3:], output_shape)

    else:
        x = tf.expand_dims(x, axis=-1)
        x = approximate_identity(x, (-1,) + output_shape + (1,))
        x = tf.squeeze(x, axis=-1)

    return x

def _approximate_identity_nd_old(x, input_shape, output_shape):
    expansion_shape = []
    compression_shape = []

    N = 5
    for a, b in zip(input_shape, output_shape):
        num, den = closest_fraction(b / a, N)
        expansion_shape.append(num)
        compression_shape.append(den)

    # TODO: implement expansion. I'm not sure about memory cost of doing it, and I do not need it immediately.
    # Ideally, I would use multi-linear interpolation.

    # if np.prod(expansion_shape) > 1:
    #     raise NotImplementedError

    if np.prod(compression_shape) > 1:
        x = tf.nn.pool(
            tf.expand_dims(x, axis=-1),
            compression_shape,
            pooling_type='AVG',
            padding='SAME',
            strides=compression_shape)
        x = tf.squeeze(x, axis=-1)

    xshape = tuple(x.shape.as_list()[1:])
    if xshape != output_shape:
        padding = tf.constant([[0, 0]] + [[0, d2 - d1] for (d1, d2) in zip(xshape, output_shape)])
        x = tf.pad(x, padding)

    xshape = tuple(x.shape.as_list()[1:])
    assert(xshape == output_shape)

    return x

def approximate_identity(x, shape):
    """Produces a tensor of given shape from x, which is close to the identity.
       In particular, if x.shape = shape, it is exactly the identity."""
    input_shape = tuple(x.shape.as_list()[1:])
    output_shape = tuple(shape)[1:]

    assert(len(input_shape) == len(output_shape))

    if input_shape == output_shape:
        y = x

    elif (len(input_shape) == 2):
        y = _approximate_identity_1d(x, input_shape, output_shape)

    elif (len(input_shape) == 3):
        y = _approximate_identity_2d(x, input_shape, output_shape)

    elif (len(input_shape) > 3):
        y = _approximate_identity_nd(x, input_shape, output_shape)

    else:
        raise NotImplementedError

    assert(y.shape[1:] == output_shape)

    return y
