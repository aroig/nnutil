import tensorflow as tf
import numpy as np


class PartitionDataset(tf.data.Dataset):
    """ Partitions the dataset pseudo-randomly into disjoint classes with relative frequencies
        given by dist. The partition is deterministic, and obtained by hashing the value given
        by key_fn applied to the features dict."""

    def __init__(self, dataset, dist, key_fn, index):
        self._nbuckets = len(dist)

        ss = sum(dist)
        self._thresh_A = sum([p for i, p in enumerate(dist) if i < index ]) / ss
        self._thresh_B = sum([p for i, p in enumerate(dist) if i <= index ]) / ss

        self._key_fn = key_fn
        self._salt = ""

        dataset = tf.data.Dataset.filter(dataset, self.has_matching_bucket)

        self._dataset = dataset


    def has_matching_bucket(self, *args):
        feature = args[0]

        path = self._key_fn(args)
        salted_path = tf.string_join([path, tf.constant(self._salt, dtype=tf.string)])

        N = 100
        bucket = tf.cast(tf.string_to_hash_bucket_fast(salted_path, N), dtype=tf.float32)
        bucket = bucket / tf.constant(N, dtype=tf.float32)

        C0 = tf.less(self._thresh_A, bucket)
        C1 = tf.less_equal(bucket, self._thresh_B)

        return tf.logical_and(C0, C1)


    def _as_variant_tensor(self):
        return self._dataset._as_variant_tensor()


    @property
    def output_classes(self):
        return self._dataset.output_classes


    @property
    def output_shapes(self):
        return self._dataset.output_shapes


    @property
    def output_types(self):
        return self._dataset.output_types


def partition(dataset, dist, key_fn):
    return tuple([PartitionDataset(dataset, dist, key_fn, i) for i in range(0, len(dist))])

