import tensorflow as tf
import tensorboard as tb
import numpy as np

from tensorboard.plugins.histogram.metadata import create_summary_metadata

def layers(name, layers=None, gradients=None, activations=None):
    def entry(i, x):
        return tf.stack([
            tf.constant(i-0.5, shape=(), dtype=tf.float32),
            tf.constant(i+0.5, shape=(), dtype=tf.float32),
            tf.cast(x, dtype=tf.float32)
        ])

    with tf.name_scope(name):
        summary_list = []

        # weight size summary
        if layers is not None:
            sizes_entries = []
            for i, l in enumerate(layers):
                size = sum([tf.reduce_prod(tf.shape(v)) for v in l.variables])
                sizes_entries.append(entry(i, size))

            sizes_tensor = tf.stack(sizes_entries)
            total_size = np.sum([np.prod(v.shape) for l in layers for v in l.variables])
            metadata = create_summary_metadata(display_name='sizes',
                                               description='total: {}'.format(total_size))
            sizes_summary = tf.summary.tensor_summary('sizes',
                                                      sizes_tensor,
                                                      summary_metadata=metadata)
            summary_list.append(sizes_summary)

        # weight L2 norm summary
        if layers is not None:
            vars_entries = []
            for i, l in enumerate(layers):
                vnorm = sum([tf.nn.l2_loss(v) for v in l.variables])
                vars_entries.append(entry(i, vnorm))

            vars_tensor = tf.stack(vars_entries)
            metadata = create_summary_metadata(display_name='weights', description=None)
            vars_summary = tf.summary.tensor_summary('weights',
                                                     vars_tensor,
                                                     summary_metadata=metadata)
            summary_list.append(vars_summary)

        # activation L2 norm summary
        if activations is not None:
            act_entries = []
            for i, x in enumerate(activations):
                xnorm = tf.nn.l2_loss(x)
                act_entries.append(entry(i, xnorm))

            act_tensor = tf.stack(act_entries)
            metadata = create_summary_metadata(display_name='activations', description=None)
            act_summary = tf.summary.tensor_summary('activations',
                                                     act_tensor,
                                                     summary_metadata=metadata)
            summary_list.append(act_summary)

        # gradient L2 norm summary
        if layers is not None and gradients is not None:
            grads_entries = []
            for i, l in enumerate(layers):
                glist = [g for g, v in gradients if v in set(l.variables)]

                gnorm = sum([tf.nn.l2_loss(g) for g in glist])
                grads_entries.append(entry(i, gnorm))

            grads_tensor = tf.stack(grads_entries)
            metadata = create_summary_metadata(display_name='gradients', description=None)
            grads_summary = tf.summary.tensor_summary('gradients',
                                                      grads_tensor,
                                                      summary_metadata=metadata)
            summary_list.append(grads_summary)

        merged_summary = tf.summary.merge(summary_list)

    return merged_summary
