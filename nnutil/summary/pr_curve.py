import tensorflow as tf
import tensorboard as tb

def pr_curve(name, probabilities, labels, label_names=None, streaming=False):
    if label_names is None:
        label_names = ['{}'.format(i) for i in range(0, labels.shape[-1])]

    summary_list = []
    with tf.name_scope(name):
        for i, lb in enumerate(label_names):
            if streaming:
                summary, update_op = tb.summary.pr_curve_streaming_op(
                    lb,
                    predictions=probabilities[:, i],
                    labels=tf.cast(tf.equal(labels, i), tf.bool),
                    num_thresholds=200,
                    metrics_collections=[tf.GraphKeys.SUMMARIES],
                    updates_collections=[tf.GraphKeys.UPDATE_OPS])
            else:
                summary = tb.summary.pr_curve_streaming_op(
                    lb,
                    predictions=probabilities[:, i],
                    labels=tf.cast(tf.equal(labels, i), tf.bool),
                    num_thresholds=200)

            summary_list.append(summary)

        merged_summary = tf.summary.merge(summary_list)

    return merged_summary
