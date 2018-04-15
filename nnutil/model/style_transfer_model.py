import os

import numpy as np
import tensorflow as tf

from .. import summary
from .. import layers
from .. import train
from .. import util

from .base_model import BaseModel

class StyleTransferModel(BaseModel):
    def __init__(self, name, shape):
        super().__init__(name)

        self._shape = shape
        self._transformer = None
        self._input_classifier = None
        self._synth_classifier = None

    @property
    def input_shape(self):
        return self._shape

    @property
    def output_shape(self):
        return self._shape

    @property
    def layers(self):
        return self._transformer.layers

    def features_placeholder(self, batch_size=1):
        return {
            'image': tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size,) + self._shape,
                                    name='image'),
        }

    def training_estimator_spec(self, loss, image, style, synth, params, config):
        step = tf.train.get_global_step()
        learning_rate = params.get('learning_rate', 0.001)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Manually apply gradients. We want the gradients for summaries.
        # We need to apply them manually in order to avoid having
        # duplicate gradient ops.
        gradients = optimizer.compute_gradients(loss)

        # Make sure we run update averages on each training step
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            train_op = optimizer.apply_gradients(gradients, global_step=step)

        summary.layers("layer_summary_{}".format(self._transformer.name),
                          self._transformer.layers,
                          gradients)

        training_hooks = []

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            training_chief_hooks=training_hooks,
            train_op=train_op
        )

    def evaluation_estimator_spec(self, loss, image, style, synth, params, config):
        evaluation_hooks = []

        eval_metric_ops = {}

        # Make sure we run update averages on each training step
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            loss = tf.identity(loss)

        evaluation_hooks.append(
            train.EvalSummarySaverHook(
                output_dir=os.path.join(config.model_dir, "eval"),
                summary_op=tf.summary.merge_all()
            )
        )

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            evaluation_hooks=evaluation_hooks,
            eval_metric_ops=eval_metric_ops
        )

    def prediction_estimator_spec(self, synth, params, config):
        predictions = {
            "synth": synth
        }

        exports = {
        }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs=exports
        )

    def style_loss(self):
        input_activations = self._input_classifier.layer_activations
        synth_activations = self._synth_classifier.layer_activations

        def gram(x):
            shape=tf.shape(x)
            xx = tf.expand_dims(x, -1)
            return tf.reshape(tf.matmul(xx, xx, transpose_a=True), shape=shape[:-1])

        loss = tf.constant(0, dtype=tf.float32)
        for i in [1, 2, 3]:
            shape = tf.shape(input_activations[i])
            input_gram = gram(input_activations[i])
            synth_gram = gram(synth_activations[i])
            U = tf.cast(tf.reduce_prod(shape[1:]), dtype=tf.float32)
            loss += tf.nn.l2_loss(input_gram - synth_gram) / U

        return loss

    def content_loss(self):
        input_activations = self._input_classifier.layer_activations
        synth_activations = self._synth_classifier.layer_activations

        loss = tf.constant(0, dtype=tf.float32)
        for i in [-1]:
            shape = tf.shape(input_activations[i])
            U = tf.cast(tf.reduce_prod(shape[1:]), dtype=tf.float32)
            loss += tf.nn.l2_loss(input_activations[i] - input_activations[i]) / U

        return loss

    def transformer_network(self, params):
        raise NotImplementedError

    def model_fn(self, features, labels, mode, params, config):
        image = features['image']
        style = features['style']
        alpha = params.get('alpha', self._alpha)

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        self._transformer = layers.Segment(self.transformer_network(params), name="transformer")

        synth = self._transformer.apply(image, training=training)
        synth = tf.tanh(synth)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.prediction_estimator_spec(synth, params, config)

        self._input_classifier = layers.Mobilenet(name="input_classifier")
        self._synth_classifier = layers.Mobilenet(name="synth_classifier")

        self._input_classifier.apply(image)
        self._synth_classifier.apply(synth)

        # Calculate total loss function
        with tf.name_scope('losses'):
            loss = alpha * self.style_loss() + (1 - alpha) * self.content_loss()
            loss += sum([l for l in self._input_classifier.losses])
            loss += sum([l for l in self._synth_classifier.losses])
            loss += sum([l for l in self._transformer.losses])

        tf.summary.image("synth", util.mosaic(tf.stack([image[0,...], synth[0,...]])))

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.training_estimator_spec(loss, image, style, synth, params, config)

        else:
            return self.evaluation_estimator_spec(loss, image, style, synth, params, config)
