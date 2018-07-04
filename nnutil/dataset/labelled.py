import tensorflow as tf
import numpy as np

from .merge import merge
from .interleave import interleave

def labelled(datasets, label_key=None, label_str_key=None, weights=None, seed=None):

    if weights is None:
        weights = {}

    merged_datasets = []
    for idx, (lb, ds) in enumerate(datasets.items()):

        label_feature = {}

        if label_key is not None:
            label_feature[label_key] = tf.constant(idx, dtype=tf.int32)

        if label_str_key is not None:
            label_feature[label_str_key] = tf.constant(lb, dtype=tf.string)

        label_ds = tf.data.Dataset.from_tensors(label_feature)

        label_ds = label_ds.repeat()

        merged_ds = merge([ds, label_ds])
        merged_ds = merged_ds.repeat()

        merged_datasets.append(merged_ds)

    sample_dataset = tf.contrib.data.sample_from_datasets(
       merged_datasets,
       weights=[weights.get(lb, 1.0) for lb in datasets.keys()],
       seed=seed)

    return sample_dataset
