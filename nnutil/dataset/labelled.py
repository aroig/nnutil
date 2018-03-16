import tensorflow as tf
import numpy as np

from .merge import merge
from .interleave import interleave

def labelled(datasets, label_key=None, label_str_key=None):
    merged_datasets = []
    for idx, (lb, ds) in enumerate(datasets.items()):

        label_feature = {}

        if label_key is not None:
            label_feature[label_key] = tf.constant(idx, dtype=tf.int32)

        if label_str_key is not None:
            label_feature[label_str_key] = tf.constant(lb, dtype=tf.string)

        label_ds = tf.data.Dataset.from_tensors(label_feature)

        label_ds = label_ds.repeat()

        merged_datasets.append(merge([ds, label_ds]))

    return interleave(merged_datasets)
