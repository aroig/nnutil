import os

import numpy as np
import tensorflow as tf

from .. import summary
from .. import layers
from .. import train
from .. import util

from .base_model import BaseModel

class ClassificationModel(BaseModel):
    def __init__(self, name, shape, labels):
        super().__init__(name)

        self._shape = shape
        self._labels = labels

        self._classifier = None

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
                                    shape=(batch_size, len(self.labels)),
                                    name='label')
        }

    def classification_metrics(self, loss, labels, logits):
        metrics = {}
        metrics['loss'] = loss

        with tf.name_scope("metrics"):
            probs = tf.reshape(tf.nn.softmax(logits), shape=(-1, len(self.labels)))
            metrics['probs'] = probs

            prediction = tf.argmax(logits, axis=1)
            metrics['prediction'] = prediction

            # Shape: (truth, prediction)
            if (labels is not None and len(prediction.shape) == 1):
                confusion = tf.confusion_matrix(labels, prediction, len(self.labels))
                confusion = tf.cast(confusion, dtype=tf.float32)
                metrics['confusion'] = confusion

        return metrics

    def classification_summaries(self, image, labels, metrics, confusion):
        prediction = metrics['prediction']
        probs = metrics['probs']

        if (len(self.labels) < 20):
            mosaic = util.confusion_mosaic(image, len(self.labels), labels, prediction)
            tf.summary.image("confusion_mosaic", mosaic)

        summary.pr_curve("prcurve", probs, labels, label_names=self.labels)

        summary.classification("classification_summary", confusion, self.labels)

    def training_estimator_spec(self, loss, image, labels, logits, params, config):
        step = tf.train.get_global_step()
        learning_rate = params.get('learning_rate', 0.001)

        ema = tf.train.ExponentialMovingAverage(decay=0.85, name="ema_train")
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Manually apply gradients. We want the gradients for summaries.
        # We need to apply them manually in order to avoid having
        # duplicate gradient ops.
        gradients = optimizer.compute_gradients(loss)

        metrics = self.classification_metrics(loss, labels, logits)
        confusion = metrics['confusion']
        prediction = metrics['prediction']
        probs = metrics['probs']

        ema_update_ops = ema.apply([confusion])
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_update_ops)

        confusion_avg = ema.average(confusion)

        # Make sure we run update averages on each training step
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            train_op = optimizer.apply_gradients(gradients, global_step=step)

        summary.activation_map("activation_summary", logits, image)

        summary.layers("layer_summary_{}".format(self._classifier.name),
                          self._classifier.layers,
                          gradients)

        self.classification_summaries(image, labels, metrics, confusion_avg)

        training_hooks = []

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            training_chief_hooks=training_hooks,
            train_op=train_op
        )

    def evaluation_estimator_spec(self, loss, image, labels, logits, params, config):
        evaluation_hooks = []

        eval_metric_ops = {}

        metrics = self.classification_metrics(loss, labels, logits)
        confusion = metrics['confusion']
        prediction = metrics['prediction']
        probs = metrics['probs']

        confusion_avg, confusion_op = tf.metrics.mean_tensor(confusion)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, confusion_op)

        # Make sure we run update averages on each training step
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            loss = tf.identity(loss)

        self.classification_summaries(image, labels, metrics, confusion_avg)

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
        prediction = tf.argmax(input=logits, axis=1)

        probs = tf.reshape(tf.nn.softmax(logits),
                           shape=(-1, len(self.labels)),
                           name="probabilities")

        prediction_dict = {
            "class": prediction,
            "probs": probs
        }

        exports = {
           'class': tf.estimator.export.ClassificationOutput(scores=probs)
        }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=prediction_dict,
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

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.prediction_estimator_spec(logits, params, config)

        # Calculate total loss function
        with tf.name_scope('losses'):
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            loss += sum([l for l in self._classifier.losses])

        # Configure the training and eval phases
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.training_estimator_spec(loss, image, labels, logits, params, config)

        else:
            return self.evaluation_estimator_spec(loss, image, labels, logits, params, config)
