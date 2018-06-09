import os

import numpy as np
import tensorflow as tf

from .. import summary
from .. import layers
from .. import train
from .. import util
from .. import image

from .base_model import BaseModel

class ClassificationModel(BaseModel):
    def __init__(self, name, shape, labels, class_weights=None):
        super().__init__(name)

        self._shape = shape
        self._labels = labels

        if class_weights is None:
            self._class_weights = [1.0] * len(labels)

        self._classifier = None

    @property
    def input_shape(self):
        return self._shape

    @property
    def output_shape(self):
        return (len(self._labels),)

    @property
    def labels(self):
        return self._labels

    @property
    def layers(self):
        return self._classifier.layers

    def features_placeholder(self, batch_size=None):
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
            metrics['probs'] = self.probabilities(logits)

            prediction = tf.argmax(logits, axis=1)
            metrics['prediction'] = prediction

            # Shape: (truth, prediction)
            if (labels is not None and len(prediction.shape) == 1):
                confusion = tf.confusion_matrix(labels, prediction, len(self.labels))
                confusion = tf.cast(confusion, dtype=tf.float32)

                confusion_mass = tf.reduce_sum(confusion)
                confusion_rel = confusion / confusion_mass

                metrics['confusion'] = confusion
                metrics['confusion_rel'] = confusion_rel

                label_freq = tf.reduce_sum(confusion, axis=-1)
                label_rel = label_freq / confusion_mass

                metrics['label_freq'] = label_freq
                metrics['label_rel'] = label_rel

                accuracy = tf.trace(confusion) / confusion_mass
                class_accuracy = tf.diag_part(confusion) / label_freq

                metrics['accuracy'] = accuracy
                metrics['class_accuracy'] = class_accuracy

        return metrics

    def classification_summaries(self, image, labels, metrics, confusion):
        shape = tuple(image.shape[1:])

        prediction = metrics['prediction']
        probs = metrics['probs']
        confusion_rel = metrics['confusion_rel']
        accuracy = metrics['accuracy']
        class_accuracy = metrics['class_accuracy']
        label_rel = metrics['label_rel']

        if (len(self.labels) < 40 and int(shape[-1]) in set([1, 3])):
            mosaic = image.confusion_mosaic(image, len(self.labels), labels, prediction)
            tf.summary.image("confusion_mosaic", mosaic)

        tf.summary.image('confusion',
                         tf.reshape(confusion_rel, shape=(1, len(self.labels), len(self.labels), 1)))

        tf.summary.scalar('accuracy', accuracy)

        with tf.name_scope("class_freq"):
            for i, lb in enumerate(self.labels):
                tf.summary.scalar(lb, tf.reduce_mean(label_rel[i]))

        with tf.name_scope("class_accuracy"):
            for i, lb in enumerate(self.labels):
                tf.summary.scalar(lb, tf.reduce_mean(class_accuracy[i]))

        with tf.name_scope("class_distribution"):
            for i, lb in enumerate(self.labels):
                summary.distribution(lb, confusion[i, :])

    def training_estimator_spec(self, loss, image, labels, logits, params, config):
        step = tf.train.get_global_step()

        max_steps = params.get('train_steps', None)
        learning_rate = params.get('learning_rate', 0.001)
        learning_rate_decay = params.get('learning_rate_decay', 1.0)

        with tf.name_scope('optimizer'):
            learning_rate = learning_rate * tf.exp(tf.cast(step, dtype=tf.float32) * tf.log(learning_rate_decay))

            ema = tf.train.ExponentialMovingAverage(decay=0.85, name="ema_train")
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # Manually apply gradients. We want the gradients for summaries.
            # We need to apply them manually in order to avoid having
            # duplicate gradient ops.
            gradients = optimizer.compute_gradients(loss)

        tf.summary.scalar('learning_rate', learning_rate)

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

        if int(image.shape[-1]) in set([1, 3]):
            summary.activation_map("activation_summary", logits, image)

        summary.layers("layer_summary_{}".format(self._classifier.name),
                       layers=self._classifier.layers,
                       gradients=gradients,
                       activations=self._classifier.layer_activations)

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
        # eval_metric_ops['confusion'] = (confusion_avg, confusion_op)

        self.classification_summaries(image, labels, metrics, confusion_avg)

        summary.pr_curve("prcurve", probs, labels, label_names=self.labels, streaming=True)

        # Make sure we run update averages on each training step
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            loss = tf.identity(loss)

        eval_dir = os.path.join(config.model_dir, "eval")
        evaluation_hooks.append(
            train.EvalSummarySaverHook(
                output_dir=eval_dir,
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

        probs = self.probabilities(logits)

        prediction_dict = {
            "class": prediction,
            "probs": probs,
            "logits": logits
        }

        exports_dict = {
           'class': tf.estimator.export.ClassificationOutput(scores=probs)
        }

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=prediction_dict,
            export_outputs=exports_dict
        )

    def probabilities(self, logits):
        return tf.reshape(
            tf.nn.softmax(logits),
            shape=(-1, len(self.labels)),
            name="probabilities")

    def loss_function(self, labels, logits):
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits, reduction=tf.losses.Reduction.NONE)
        return loss

    def weighted_loss(self, labels, logits, sample_bias=0.0):
        sample_loss = self.loss_function(labels, logits)

        mu, sigma = tf.nn.moments(sample_loss, axes=[0])
        norm_sample_loss = (sample_loss - mu) / (sigma + 1e-1)

        threshold = 1.0
        sample_weights = tf.stop_gradient(tf.minimum(
            tf.exp(sample_bias * norm_sample_loss),
            threshold))

        class_weights = tf.gather_nd(
            tf.constant(self._class_weights, dtype=tf.float32),
            tf.expand_dims(labels, axis=-1))

        average_loss = tf.losses.compute_weighted_loss(
            sample_loss,
            weights=sample_weights * class_weights)

        return average_loss

    def classifier_network(self, params):
        raise NotImplementedError

    def model_fn(self, features, labels, mode, params, config):
        image = features['image']
        step = tf.train.get_global_step()

        labels = None
        if 'label' in features:
            labels = tf.cast(features['label'], tf.int32)

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        self._classifier = layers.Segment(self.classifier_network(params), name="classifier")
        logits = self._classifier.apply(image, training=training)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.prediction_estimator_spec(logits, params, config)

        # Sample weights, so that easy samples weight less
        sample_bias = params.get('sample_bias', 0.0)
        sample_bias_step = params.get('sample_bias_step', 0)

        regularizer = params.get('regularizer', 0.0)
        regularizer_step = params.get('regularizer_step', 0)

        # Calculate total loss function
        with tf.name_scope('losses'):
            if labels is not None:
                cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
                sample_bias_dampening = tf.sigmoid(tf.cast(step - sample_bias_step, dtype=tf.float32) / 10.0)
                model_loss = self.weighted_loss(labels, logits, sample_bias=sample_bias * sample_bias_dampening)

                regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                regularization_dampening = tf.sigmoid(tf.cast(step - regularizer_step, dtype=tf.float32) / 10.0)
                total_loss = model_loss + regularizer * regularization_dampening * sum([l for l in regularization_losses])

            else:
                cross_entropy = tf.constant(0, dtype=tf.float32)
                model_loss = tf.constant(0, dtype=tf.float32)
                total_loss = tf.constant(0, dtype=tf.float32)

        tf.summary.scalar("cross_entropy", cross_entropy)

        # Configure the training and eval phases
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.training_estimator_spec(total_loss, image, labels, logits, params, config)

        else:
            return self.evaluation_estimator_spec(total_loss, image, labels, logits, params, config)
