import os

import numpy as np
import tensorflow as tf

from .base_model import BaseModel
from .. import layers

class ClassificationModel(BaseModel):
    def __init__(self, name, shape, labels):
        super().__init__(name)

        self._shape = shape
        self._labels = labels

        self._nlabels = len(self._labels)
        self._classifier = None

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def labels(self):
        return self._labels

    @property
    def layers(self):
        return self._classifier.layers

    @property
    def layer_sizes(self):
        sizes = []
        for l in self._classifier.layers:
            for v in l.variables:
                sizes.append(int(np.prod(v.shape)))

        return sizes

    def features_placeholder(self, batch_size=1):
        return {
            'image': tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size,) + self._shape,
                                    name='image'),
            'label': tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size, self._nlabels),
                                    name='label')
        }

    def class_summaries(self, confusion, labels):
        # Shape: ()
        accuracy = tf.trace(confusion)

        # Shape: (nlabels)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self._nlabels)
        label_stats = tf.reduce_mean(onehot_labels, axis=0)

        # Shape: (nlabels)
        class_accuracy = tf.stack([confusion[i,i] / tf.reduce_sum(confusion[i])
                                   for i in range(0, self._nlabels)])

        tf.summary.scalar('accuracy', accuracy)
        # tf.summary.histogram('label_stats', label_stats)

        for i, lb in enumerate(self._labels):
            name = 'class_freq/{}'.format(lb)
            tf.summary.scalar(name, tf.reduce_mean(label_stats[i]))

        for i, lb in enumerate(self._labels):
            name = 'class_accuracy/{}'.format(lb)
            tf.summary.scalar(name, tf.reduce_mean(class_accuracy[i]))

    def evaluation_metrics(self, confusion, labels):
        metric_ops = {}

        # Shape: ()
        accuracy = tf.trace(confusion)
        metric_ops['accuracy'] = tf.metrics.mean(accuracy)

        # Shape: (nlabels)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self._nlabels)
        label_stats = tf.reduce_mean(onehot_labels, axis=0)

        # Shape: (nlabels)
        class_accuracy = tf.stack([confusion[i,i] / tf.reduce_sum(confusion[i])
                                   for i in range(0, self._nlabels)])

        for i, lb in enumerate(self._labels):
            name = 'class_freq/{}'.format(lb)
            metric_ops[name] = tf.metrics.mean(label_stats[i])

        for i, lb in enumerate(self._labels):
            name = 'class_accuracy/{}'.format(lb)
            metric_ops[name] = tf.metrics.mean(class_accuracy[i])

        return metric_ops


    def training_estimator_spec(self, loss, confusion, labels, params, config):
        step = tf.train.get_global_step()

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        # Manually apply gradients. We want the gradients for summaries.
        # We need to apply them manually in order to avoid having
        # duplicate gradient ops.
        gradients = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gradients, global_step=step)

        self.variable_summaries(gradients)
        self.class_summaries(confusion, labels)

        training_hooks = []

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            training_chief_hooks=training_hooks,
            train_op=train_op
        )

    def evaluation_estimator_spec(self, loss, confusion, labels, params, config):
        evaluation_hooks = []

        eval_metric_ops = {}

        self.class_summaries(confusion, labels)

        evaluation_hooks.append(
            tf.train.SummarySaverHook(
                save_steps=params['eval_steps'],
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

    def classifier_network(self):
        raise NotImplementedError

    def model_fn(self, features, labels, mode, params, config):
        image = features['image']
        labels = tf.cast(features['label'], tf.int32)

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        self._classifier = layers.Segment(layers=self.classifier_network(), name="classifier")
        logits = self._classifier.apply(image, training=training)

        predicted_class = tf.argmax(input=logits, axis=1)

        # Shape: (nlabels, nlabels)
        confusion = tf.confusion_matrix(labels, predicted_class, self._nlabels)
        confusion = tf.cast(confusion, dtype=tf.float32)
        confusion = confusion / tf.reduce_sum(confusion)

        tf.summary.image('confusion', tf.expand_dims(tf.expand_dims(confusion, axis=-3), axis=-1))

        # Orientation test
        # confusion = tf.constant([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=tf.float32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.prediction_estimator_spec(logits, params, config)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        tf.summary.scalar('loss', loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.training_estimator_spec(loss, confusion, labels, params, config)

        else:
            return self.evaluation_estimator_spec(loss, confusion, labels, params, config)
