import os
import tensorflow as tf

from .labelled import labelled
from .image_files import image_files

def labelled_images(path, shape, glob='*', labels=None, tfrecord=False):
    path = os.path.abspath(path)

    if labels is None:
        labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    def parse_example(example_proto):
        features = {
            "path": tf.FixedLenFeature((), tf.string),
            "shape": tf.FixedLenFeature((3,), tf.int64),
            "image": tf.VarLenFeature(tf.float32),
            "label": tf.FixedLenFeature((), tf.int64)
        }

        parsed_features = tf.parse_single_example(example_proto, features)

        image = tf.sparse_tensor_to_dense(parsed_features['image'], default_value=0)
        image = tf.reshape(image, shape=parsed_features['shape'])

        image = tf.image.resize_images(image, size=shape[0:2])

        if shape[2] == 1:
            image = tf.cond(tf.equal(parsed_features['shape'][2], 1),
                            lambda: image,
                            lambda: tf.image.rgb_to_grayscale(image))

        elif shape[2] == 3:
            image = tf.cond(tf.equal(parsed_features['shape'][2], 3),
                            lambda: image,
                            lambda: tf.image.grayscale_to_rgb(image))

        parsed_features['image'] = tf.reshape(image, shape=shape)

        return parsed_features

    if tfrecord:
        dataset = tf.data.TFRecordDataset(filenames=[path + '.tfrecord'])
        dataset = dataset.map(parse_example)
        # TODO: filter labels using the regexp op in 1.7

    else:
        dataset = labelled({
            lb: image_files(os.path.join(path, lb),
                            shape,
                            glob=glob) for lb in labels
        }, label_key='label')

    return dataset
