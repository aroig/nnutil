import tensorflow as tf

def activation_map(name, logits, image):
    grads = tf.gradients(tf.reduce_max(logits, axis=1), image)[0]

    # pick first element in the batch
    heatmap = grads[0]
    image = image[0]

    heatmap = tf.abs(heatmap)
    heatmap = heatmap / tf.reduce_max(heatmap)

    masked_image = tf.concat([tf.expand_dims(image, 0),
                              tf.expand_dims(heatmap, 0)], axis=2)

    return tf.summary.image(name, masked_image)
