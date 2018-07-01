import tensorflow as tf
import numpy as np

def as_generator(dataset):
    with tf.Session() as sess:
        it = dataset.make_one_shot_iterator()
        feature = it.get_next()
        try:
            while True:
                feature_np = sess.run(feature)
                yield feature_np

        except tf.errors.OutOfRangeError:
            pass
