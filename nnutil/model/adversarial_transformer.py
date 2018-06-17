import os

import numpy as np
import tensorflow as tf
import nnutil as nn

from .base_model import BaseModel


class AdversarialTransformer(BaseModel):
    def __init__(self, name, shape):
        super(AdversarialTransformer, self).__init__(name)

        self._shape = shape

        self._transformer = None
        self._discriminator = None

    @property
    def input_shape(self):
        return self._shape

    @property
    def output_shape(self):
        return self._shape

    @property
    def layers(self):
        return self._transformer.layers

    def transformer_network(self, params):
        raise NotImplementedError

    def discriminator_network(self, params):
        raise NotImplementedError

    def features_placeholder(self, batch_size=1):
        return {
            'source': tf.placeholder(dtype=tf.float32,
                                     shape=(batch_size,) + self._shape,
                                     name='source'),
            'target': tf.placeholder(dtype=tf.float32,
                                     shape=(batch_size,) + self._shape,
                                     name='target')
        }

    def loss_function(self, tgt_image, synth_image, params):
        step = tf.train.get_global_step()

        # Sample weights, so that easy samples weight less
        sample_bias = params.get('sample_bias', 0.0)
        sample_bias_step = params.get('sample_bias_step', 0)

        # Regularizer weight
        regularizer = params.get('regularizer', 0.0)
        regularizer_step = params.get('regularizer_step', 0)

        # Calculate total loss function
        with tf.name_scope('losses'):
            sample_loss = tf.norm(nn.util.flatten(synth_image - tgt_image), ord=2, axis=1)

            # TODO: perform importance sampling here

            model_loss = tf.reduce_mean(sample_loss)

            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_dampening = tf.sigmoid(tf.cast(step - regularizer_step, dtype=tf.float32) / 10.0)
            total_loss = model_loss + regularizer * regularization_dampening * sum([l for l in regularization_losses])

        tf.summary.scalar("model_loss", model_loss)

        return total_loss

    def model_fn(self, features, labels, mode, params, config):
        src_image = features['source']
        tgt_image = features['target']
        step = tf.train.get_global_step()

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        self._transformer = nn.layers.Segment(self.transformer_network(params), name="transformer")
        self._discriminator = nn.layers.Segment(self.transformer_network(params), name="discriminator")

        synth_image = self._transformer.apply(src_image, training=training)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.prediction_estimator_spec(src_image, synth_image, params, config)

        loss = self.loss_function(tgt_image, synth_image, params)

        # Configure the training and eval phases
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.training_estimator_spec(loss, src_image, synth_image, tgt_image, params, config)

        else:
            return self.evaluation_estimator_spec(loss, src_image, synth_image, tgt_image, params, config)

    def training_estimator_spec(self, loss, src_image, synth_image, tgt_image, params, config):
        step = tf.train.get_global_step()
        learning_rate = params.get('learning_rate', 0.0001)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0, beta2=0.9)

        # Manually apply gradients. We want the gradients for summaries. We need
        # to apply them manually in order to avoid having duplicate gradient ops.
        gradients = optimizer.compute_gradients(loss)

        # Make sure we update averages on each training step
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            train_op = optimizer.apply_gradients(gradients, global_step=step)

        nn.summary.image_transformation(
            "transformation",
            src_image[0, :],
            synth_image[0, :])

        nn.summary.image_transformation(
            "truth",
            tgt_image[0, :],
            synth_image[0, :])

        nn.summary.layers("layer_summary_{}".format(self._transformer.name),
                          layers=self._transformer.layers,
                          gradients=gradients,
                          activations=self._transformer.layer_activations)

        nn.summary.layers("layer_summary_{}".format(self._discriminator.name),
                          layers=self._discriminator.layers,
                          gradients=gradients,
                          activations=self._discriminator.layer_activations)

        training_hooks = []

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            training_hooks=training_hooks,
            train_op=train_op)

    def evaluation_estimator_spec(self, loss, src_image, synth_image, tgt_image, params, config):
        eval_metric_ops = {}

        evaluation_hooks = []

        # Make sure we run update averages on each training step
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            loss = tf.identity(loss)

        eval_dir = os.path.join(config.model_dir, "eval")
        evaluation_hooks.append(
            nn.train.EvalSummarySaverHook(
                output_dir=eval_dir,
                summary_op=tf.summary.merge_all()
            )
        )

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            evaluation_hooks=evaluation_hooks,
            eval_metric_ops=eval_metric_ops)

    def prediction_estimator_spec(self, src_image, synth_image, params, config):
        predictions = {
            "synth": synth_image
        }

        exports = {}

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs=exports)
