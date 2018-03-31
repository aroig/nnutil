import os

import numpy as np
import tensorflow as tf
import nnutil as nn

from .base_model import BaseModel

class WGANModel(BaseModel):
    def __init__(self, name, shape, code_size, autoencoder=False):
        super(WGANModel, self).__init__(name)

        self._shape = shape
        self._code_shape = (code_size,)
        self._autoencoder = autoencoder

        self._encoder = None
        self._generator = None
        self._critic = None

    @property
    def shape(self):
        return self._shape

    def features_placeholder(self, batch_size=1):
        return {
            'image': tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size,) + self._shape,
                                    name='image')
        }

    def autoencoder_summaries(self, image, code, synthetic):
        if self._autoencoder:
            tf.summary.image('sample', tf.concat([tf.expand_dims(image[0,...], 0),
                                                  tf.expand_dims(synthetic[0,...], 0)], axis=2))

        else:
            tf.summary.image('sample', tf.expand_dims(synthetic[0,...], 0))



    def training_estimator_spec(self, loss, image, code, synthetic, params, config):
        step = tf.train.get_global_step()

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        # Manually apply gradients. We want the gradients for summaries. We need
        # to apply them manually in order to avoid having duplicate gradient ops.
        gradients = optimizer.compute_gradients(loss)

        # Make sure we run update averages on each training step
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            train_op = optimizer.apply_gradients(gradients, global_step=step)

        self.variable_summaries(gradients)
        self.autoencoder_summaries(image, code, synthetic)

        training_hooks = []

        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                          loss=loss,
                                          training_hooks=training_hooks,
                                          train_op=train_op)

    def evaluation_estimator_spec(self, loss, image, code, synthetic, params, config):
        eval_metric_ops = {}

        evaluation_hooks = []

        # Make sure we run update averages on each training step
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            loss = tf.identity(loss)

        self.autoencoder_summaries(image, code, synthetic)

        evaluation_hooks.append(
            nn.train.EvalSummarySaverHook(
                output_dir=os.path.join(config.model_dir, "eval"),
                summary_op=tf.summary.merge_all()
            )
        )

        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                          loss=loss,
                                          evaluation_hooks=evaluation_hooks,
                                          eval_metric_ops=eval_metric_ops)

    def prediction_estimator_spec(self, image, code, synthetic, params, config):
        predictions = {
            "synthetic": synthetic
        }

        exports = {}

        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                          predictions=predictions,
                                          export_outputs=exports)

    def encoder_network(self):
        raise NotImplementedError

    def generative_network(self):
        raise NotImplementedError

    def critic_network(self):
        raise NotImplementedError

    def model_fn(self, features, labels, mode, params, config):
        image = features['image']
        batch_size = tf.shape(image)[0]

        training=(mode==tf.estimator.ModeKeys.TRAIN)

        # Generator
        self._generator = nn.layers.Segment(layers=self.generative_network(), name="generator")

        code = tf.random_uniform(shape=(batch_size,) + self._code_shape,
                                 minval=-1., maxval=1., dtype=tf.float32)

        synthetic = tf.nn.sigmoid(self._generator.apply(code, training=training))
        synthetic_ng = tf.stop_gradient(synthetic)

        # Critic
        self._critic = nn.layers.Segment(layers=self.critic_network(), name="critic")

        f_synth = self._critic.apply(synthetic, training=training)
        f_synth_ng = self._critic.apply(synthetic_ng, training=training)
        f_data = self._critic.apply(image, training=training)

        # Autoencoder
        if self._autoencoder:
            self._encoder = nn.layers.Segment(layers=self.encoder_network(), name="encoder")

            code_ae = self._encoder.apply(synthetic, training=training)

        # Losses
        loss_wgan = tf.reduce_mean(f_data - f_synth)

        loss_ae = tf.constant(0, dtype=tf.float32)
        if self._autoencoder:
            loss_ae = tf.nn.l2_loss(code - code_ae) / tf.cast(batch_size, dtype=tf.float32)

        loss_crit = -tf.reduce_mean(f_data - f_synth_ng)

        loss_lip = sum([tf.square(tf.nn.relu(tf.nn.l2_loss(w) - 2))
                        for l in self._critic.layers for w in l.variables])

        alpha = tf.exp(-tf.stop_gradient(loss_lip))
        loss = alpha * (0.01 * loss_wgan + loss_crit) + loss_lip

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.prediction_estimator_spec(image, code, synthetic, params, config)

        tf.summary.scalar('loss/wgan', loss_wgan)
        tf.summary.scalar('loss/lip', loss_lip)
        tf.summary.scalar('loss/ae', loss_ae)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.training_estimator_spec(loss, image, code, synthetic, params, config)

        else:
            return self.evaluation_estimator_spec(loss, image, code, synthetic, params, config)
