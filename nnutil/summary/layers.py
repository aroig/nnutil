
import tensorflow as tf
import tensorboard as tb

from tensorboard.plugins.histogram.metadata import create_summary_metadata

def layers(name, layers, gradients):
    vars_entries = []
    grads_entries = []
    sizes_entries = []

    def entry(i, x):
        return tf.stack([
            tf.constant(i-0.5, shape=(), dtype=tf.float32),
            tf.constant(i+0.5, shape=(), dtype=tf.float32),
            tf.cast(x, dtype=tf.float32)
        ])

    for i, l in enumerate(layers):
        glist = [g for g, v in gradients if v in set(l.variables)]
        vlist = [v for g, v in gradients if v in set(l.variables)]

        vnorm = sum([tf.nn.l2_loss(v) for v in vlist])
        vars_entries.append(entry(i, vnorm))

        gnorm = sum([tf.nn.l2_loss(g) for g in glist])
        grads_entries.append(entry(i, gnorm))

        size = sum([tf.reduce_prod(tf.shape(v)) for v in vlist])
        sizes_entries.append(entry(i, size))

    metadata = create_summary_metadata(display_name=None, description=None)

    tensor = tf.stack(grads_entries)
    grads_summary = tf.summary.tensor_summary('{}/grads'.format(name), tensor, summary_metadata=metadata)

    tensor = tf.stack(vars_entries)
    vars_summary = tf.summary.tensor_summary('{}/vars'.format(name), tensor, summary_metadata=metadata)

    tensor = tf.stack(sizes_entries)
    sizes_summary = tf.summary.tensor_summary('{}/size'.format(name), tensor, summary_metadata=metadata)

    return tf.summary.merge([vars_summary, grads_summary, sizes_summary])
