import tensorflow as tf

def activation_map(name, logits, image, family=None):
    with tf.name_scope(name):
        grads = tf.gradients(tf.reduce_max(logits, axis=1), image)[0]

        # pick first element in the batch
        heatmap = grads[0]
        image = image[0]

        heatmap = tf.abs(heatmap)
        heatmap = heatmap / tf.reduce_max(heatmap)

        heatmap = tf.image.convert_image_dtype(heatmap, dtype=tf.uint8, saturate=True)
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

        pair_image = tf.concat([tf.expand_dims(image, 0),
                                tf.expand_dims(heatmap, 0)], axis=2)

        summary = tf.summary.image('map', pair_image, family=family)

    return summary
