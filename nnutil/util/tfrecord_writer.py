import sys
import tensorflow as tf
import numpy as np

class TFRecordWriter:
    def __init__(self, path):
        self._path = path

    def __enter__(self):
        self._writer = tf.python_io.TFRecordWriter(self._path)
        return self

    def __exit__(self, type, value, traceback):
        self._writer.close()

    def make_feature(self, feature):
        if type(feature) == dict:
            return tf.train.Features(feature={k: self.make_feature(v) for k, v in feature.items()})

        elif type(feature) == bytes:
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))

        elif type(feature) in set([float, np.float32, np.float64]):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[feature]))

        elif type(feature) in set([int, np.int32, np.int64]):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[feature]))

        elif type(feature) in set([np.array, np.ndarray]):
            flat = feature.flatten().tolist()
            if feature.dtype == int or feature.dtype == np.int32 or feature.dtype == np.int64:
                return tf.train.Feature(int64_list=tf.train.Int64List(value=flat))

            elif feature.dtype == float or feature.dtype == np.float32 or feature.dtype == np.float64:
                return tf.train.Feature(float_list=tf.train.FloatList(value=flat))

            else:
                raise Exception("Unhandled array type: {}".format(feature.dtype))

        else:
            raise Exception("Unhandled feature type: {}".format(type(feature)))


    def write(self, dataset):
        dataset = dataset.prefetch(buffer_size=100)

        with tf.Session() as sess:
            it = dataset.make_one_shot_iterator()
            x = it.get_next()

            try:
                count = 0
                while True:
                    if (count % 100 == 0):
                        sys.stdout.write("Progress: {}\r".format(count))
                    count = count + 1

                    feature = sess.run([x])
                    example = tf.train.Example(features=self.make_feature(feature[0]))
                    self._writer.write(example.SerializeToString())

            except tf.errors.OutOfRangeError:
                return


def tfrecord_writer(path):
    return TFRecordWriter(path)
