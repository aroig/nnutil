import tensorflow as tf

class ImageNormalization(tf.layers.Layer):
    def __init__(self, **kwargs):
        super(ImageNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        outputs = tf.map_fn(lambda x: tf.image.per_image_standardization(x), inputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
