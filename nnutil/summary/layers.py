import tensorflow as tf
import tensorboard as tb
import numpy as np

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

    with tf.name_scope(name):
        for i, l in enumerate(layers):
            glist = [g for g, v in gradients if v in set(l.variables)]
            vlist = [v for g, v in gradients if v in set(l.variables)]

            vnorm = sum([tf.nn.l2_loss(v) for v in vlist])
            vars_entries.append(entry(i, vnorm))
            vars_tensor = tf.stack(vars_entries)

            gnorm = sum([tf.nn.l2_loss(g) for g in glist])
            grads_entries.append(entry(i, gnorm))
            grads_tensor = tf.stack(grads_entries)

            size = sum([tf.reduce_prod(tf.shape(v)) for v in vlist])
            sizes_entries.append(entry(i, size))
            sizes_tensor = tf.stack(sizes_entries)

        metadata = create_summary_metadata(display_name='grads', description=None)
        grads_summary = tf.summary.tensor_summary('grads',
                                                  grads_tensor,
                                                  summary_metadata=metadata)

        metadata = create_summary_metadata(display_name='vars', description=None)
        vars_summary = tf.summary.tensor_summary('vars',
                                                 vars_tensor,
                                                 summary_metadata=metadata)

        total_size = np.sum([np.prod(v.shape) for l in layers for v in l.variables])
        metadata = create_summary_metadata(display_name='sizes',
                                           description='total: {}'.format(total_size))
        sizes_summary = tf.summary.tensor_summary('sizes',
                                                  sizes_tensor,
                                                  summary_metadata=metadata)

        merged_summary = tf.summary.merge([vars_summary,
                                           grads_summary,
                                           sizes_summary])

    return merged_summary
