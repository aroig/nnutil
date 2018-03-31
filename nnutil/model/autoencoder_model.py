import numpy as np
import tensorflow as tf

from .base_model import BaseModel
from .. import layers

class AutoencoderModel(BaseModel):
    def __init__(self, name, shape, code_size):
        super(AutoencoderModel, self).__init__(name)

        self._shape = shape
        self._code_shape = (code_size,)

        self._encoder = None
        self._decoder = None

    @property
    def shape(self):
        return self._shape

    def features_placeholder(self, batch_size=1):
        return {
            'image': tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size,) + self._shape,
                                    name='image')
        }

    def training_estimator_spec(self, loss):
        step = tf.train.get_global_step()

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        # Manually apply gradients. We want the gradients for summaries.
        # We need to apply them manually in order to avoid having
        # duplicate gradient ops.
        gradients = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gradients, global_step=step)

        self.variable_summaries(gradients)

        training_hooks = []

        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                          loss=loss,
                                          training_hooks=training_hooks,
                                          train_op=train_op)

    def evaluation_estimator_spec(self, loss):
        eval_metric_ops = {}

        evaluation_hooks = []

        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                          loss=loss,
                                          evaluation_hooks=evaluation_hooks,
                                          eval_metric_ops=eval_metric_ops)

    def prediction_estimator_spec(self, code):
        predictions = {
            "code": code
        }

        exports = {}

        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                          predictions=predictions,
                                          export_outputs=exports)

    def encoder_network(self):
        raise NotImplementedError

    def decoder_network(self):
        raise NotImplementedError

    def model_fn(self, features, labels, mode):
        image = features['image']
        training = (mode == tf.estimator.ModeKeys.TRAIN)

        self._decoder = layers.Segment(layers=self.decoder_network(), name="decoder")
        self._encoder = layers.Segment(layers=self.encoder_network(), name="encoder")

        code = self._encoder.apply(image, training=training)
        logits = self._decoder.apply(code, training=training)
        synthetic = tf.sigmoid(logits)

        tf.summary.image('sample', tf.concat([tf.expand_dims(image[0,...], 0),
                                              tf.expand_dims(synthetic[0,...], 0)], axis=2))

        flat_shape = (-1, np.prod(image.shape[1:]))
        image_flat = tf.reshape(image, shape=flat_shape)
        logits_flat = tf.reshape(logits, shape=flat_shape)
        synthetic_flat = tf.reshape(synthetic, shape=flat_shape)

        loss_xentropy = tf.losses.sigmoid_cross_entropy(image_flat, logits_flat)

        loss_l2 = tf.losses.mean_squared_error(image_flat, synthetic_flat)

        # loss = loss_xentropy
        loss = loss_l2

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.prediction_estimator_spec(code)

        # Calculate Loss (for both TRAIN and EVAL modes)
        tf.summary.scalar('loss/l2', loss_l2)
        tf.summary.scalar('loss/xentropy', loss_xentropy)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.training_estimator_spec(loss)

        else:
            return self.evaluation_estimator_spec(loss)
