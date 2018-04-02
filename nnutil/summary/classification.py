import tensorflow as tf
import tensorboard as tb

def classification(name, confusion, label_names):
    with tf.name_scope(name):
        confusion = tf.cast(confusion, dtype=tf.float32)
        confusion_mass = tf.reduce_sum(confusion)

        label_freq = tf.reduce_sum(confusion, axis=-1)
        label_rel = label_freq / confusion_mass

        accuracy = tf.trace(confusion) / confusion_mass
        class_accuracy = tf.diag_part(confusion) / label_freq

        confusion_img = tf.expand_dims(
            tf.expand_dims(
                confusion / confusion_mass,
                axis=-3),
            axis=-1)

        tf.summary.image('confusion', confusion_img)
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('{}_class'.format(name)):
        for i, lb in enumerate(label_names):
            tf.summary.scalar(lb, tf.reduce_mean(label_rel[i]))

    with tf.name_scope('{}_freq'.format(name)):
        for i, lb in enumerate(label_names):
            tf.summary.scalar(lb, tf.reduce_mean(class_accuracy[i]))
