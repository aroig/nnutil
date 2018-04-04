import os

import numpy as np
import tensorflow as tf

from .. import summary
from .. import layers
from .. import train

from .base_model import BaseModel

class ClassificationModel(BaseModel):
    def __init__(self, name, shape, labels):
        super().__init__(name)

        self._shape = shape
        self._labels = labels

        self._nlabels = len(self._labels)
        self._classifier = None

        self._learning_rate = 0.001

    @property
    def shape(self):
        return self._shape

    @property
    def labels(self):
        return self._labels

    @property
    def layers(self):
        return self._classifier.layers

    def features_placeholder(self, batch_size=1):
        return {
            'image': tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size,) + self._shape,
                                    name='image'),
            'label': tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size, self._nlabels),
                                    name='label')
        }

    def training_estimator_spec(self, loss, confusion, params, config):
        step = tf.train.get_global_step()
        learning_rate = params.get('learning_rate', 0.001)

        ema = tf.train.ExponentialMovingAverage(decay=0.85, name="ema_train")
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Manually apply gradients. We want the gradients for summaries.
        # We need to apply them manually in order to avoid having
        # duplicate gradient ops.
        gradients = optimizer.compute_gradients(loss)

        ema_update_ops = ema.apply([confusion])
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_update_ops)

        confusion_avg = ema.average(confusion)

        # Make sure we run update averages on each training step
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            train_op = optimizer.apply_gradients(gradients, global_step=step)

        summary.layers("layer_summary_{}".format(self._classifier.name),
                          self._classifier.layers,
                          gradients)

        summary.classification("classification_summary", confusion_avg, self._labels)

        training_hooks = []

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            training_chief_hooks=training_hooks,
            train_op=train_op
        )

    def evaluation_estimator_spec(self, loss, confusion, params, config):
        evaluation_hooks = []

        eval_metric_ops = {}

        confusion_avg, confusion_op = tf.metrics.mean_tensor(confusion)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, confusion_op)

        # Make sure we run update averages on each training step
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            loss = tf.identity(loss)

        summary.classification("classification_summary", confusion_avg, self._labels)

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

    def prediction_estimator_spec(self, logits, params, config):
        predicted_class = tf.argmax(input=logits, axis=1)

        probabilities = tf.reshape(tf.nn.softmax(logits),
                                   shape=(-1, self._nlabels),
                                   name="probabilities")

        predictions = {
            "class": predicted_class,
            "probs": probabilities
        }

        exports = {
           'class': tf.estimator.export.ClassificationOutput(scores=probabilities)
        }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs=exports
        )

    def classifier_network(self, params):
        raise NotImplementedError

    def model_fn(self, features, labels, mode, params, config):
        image = features['image']
        labels = tf.cast(features['label'], tf.int32)

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        self._classifier = layers.Segment(self.classifier_network(params), name="classifier")
        logits = self._classifier.apply(image, training=training)

        with tf.name_scope("prediction"):
            probabilities = tf.reshape(tf.nn.softmax(logits), shape=(-1, self._nlabels))
            predicted_class = tf.argmax(logits, axis=1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.prediction_estimator_spec(logits, params, config)

        with tf.name_scope("confusion"):
            # Shape: (nlabels pred, nlabels truth)
            confusion = tf.confusion_matrix(labels, predicted_class, self._nlabels)
            confusion = tf.cast(confusion, dtype=tf.float32)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss += sum([l for l in self._classifier.losses])

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            summary.activation_map("activation_summary", logits, image)
            return self.training_estimator_spec(loss, confusion, params, config)

        else:
            summary.pr_curve("prcurve", probabilities, labels, label_names=self._labels)
            return self.evaluation_estimator_spec(loss, confusion, params, config)
