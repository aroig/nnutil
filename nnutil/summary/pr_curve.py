import tensorflow as tf
import tensorboard as tb

def pr_curve(name, probabilities, labels, label_names=None):
    if label_names is None:
        label_names = ['{}'.format(i) for i in range(0, labels.shape[-1])]

    summary_list = []
    for i, lb in enumerate(label_names):
        summary, update_op = tb.summary.pr_curve_streaming_op(
            '{}/{}'.format(name, lb),
            predictions=probabilities[:, i],
            labels=tf.cast(tf.equal(labels, i), tf.bool),
            num_thresholds=200,
            metrics_collections=[tf.GraphKeys.SUMMARIES],
            updates_collections=[tf.GraphKeys.UPDATE_OPS])

        summary_list.append(summary)

    return tf.summary.merge(summary_list)
