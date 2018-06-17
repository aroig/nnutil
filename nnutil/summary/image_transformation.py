
import tensorflow as tf
import nnutil as nn

def image_transformation(name, image0, image1):
    image0 = nn.image.to_rgb(image0)
    image1 = nn.image.to_rgb(image1)

    if len(image0.shape) == 3:
        image0 = tf.expand_dims(image0, axis=0)

    if len(image1.shape) == 3:
        image1 = tf.expand_dims(image1, axis=0)

    image_matrix = tf.stack([
        image0, image1, tf.abs(image0 - image1)
    ])
    image_matrix = tf.transpose(image_matrix, perm=[1, 0, 2, 3, 4])

    mosaic = nn.image.mosaic(image_matrix, border=True)
    mosaic = tf.expand_dims(mosaic, axis=0)

    return tf.summary.image(name, mosaic)
