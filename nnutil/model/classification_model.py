import os

import numpy as np
import tensorflow as tf
import tensorboard as tb
import nnutil as nn

from .base_model import BaseModel

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

    def class_summaries(self, confusion, labels_freq):
        tf.summary.image('confusion', tf.expand_dims(tf.expand_dims(confusion, axis=-3), axis=-1))

        # Shape: ()
        accuracy = tf.trace(confusion)

        # Shape: (nlabels)
        class_accuracy = tf.stack([confusion[i,i] / tf.reduce_sum(confusion[i])
                                   for i in range(0, self._nlabels)])

        tf.summary.scalar('accuracy', accuracy)
        # tf.summary.histogram('label_stats', label_stats)

        for i, lb in enumerate(self._labels):
            name = 'class_freq/{}'.format(lb)
            tf.summary.scalar(name, tf.reduce_mean(labels_freq[i]))

        for i, lb in enumerate(self._labels):
            name = 'class_accuracy/{}'.format(lb)
            tf.summary.scalar(name, tf.reduce_mean(class_accuracy[i]))

    def pr_curve_metric(self, probabilities, labels):
        for i, lb in enumerate(self._labels):
            summary, update_op = tb.summary.pr_curve_streaming_op(
                'pr/{}'.format(lb),
                predictions=probabilities[:, i],
                labels=tf.cast(tf.equal(labels, i), tf.bool),
                num_thresholds=200,
                metrics_collections='pr')

            tf.add_to_collection(tf.GraphKeys.SUMMARIES, summary)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)

    def training_estimator_spec(self, loss, confusion, labels_freq, params, config):
        step = tf.train.get_global_step()

        ema = tf.train.ExponentialMovingAverage(decay=0.85, name="ema_train")
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        # Manually apply gradients. We want the gradients for summaries.
        # We need to apply them manually in order to avoid having
        # duplicate gradient ops.
        gradients = optimizer.compute_gradients(loss)

        ema_update_ops = ema.apply([confusion, labels_freq])
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_update_ops)

        confusion_avg = ema.average(confusion)
        labels_freq_avg = ema.average(labels_freq)

        # Make sure we run update averages on each training step
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            train_op = optimizer.apply_gradients(gradients, global_step=step)

        self.variable_summaries(gradients)
        self.class_summaries(confusion_avg, labels_freq_avg)

        training_hooks = []

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            training_chief_hooks=training_hooks,
            train_op=train_op
        )

    def evaluation_estimator_spec(self, loss, confusion, labels_freq, params, config):
        evaluation_hooks = []

        eval_metric_ops = {}

        confusion_avg, confusion_op = tf.metrics.mean_tensor(confusion)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, confusion_op)

        labels_freq_avg, labels_freq_op = tf.metrics.mean_tensor(labels_freq)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, labels_freq_op)

        # Make sure we run update averages on each training step
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_ops):
            loss = tf.identity(loss)

        self.class_summaries(confusion_avg, labels_freq_avg)

        evaluation_hooks.append(
            nn.train.EvalSummarySaverHook(
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

        self._classifier = nn.layers.Segment(layers=self.classifier_network(), name="classifier")
        logits = self._classifier.apply(image, training=training)

        probabilities = tf.reshape(tf.nn.softmax(logits), shape=(-1, self._nlabels))

        predicted_class = tf.argmax(input=logits, axis=1)

        # Shape: (nlabels, nlabels)
        confusion = tf.confusion_matrix(labels, predicted_class, self._nlabels)
        confusion = tf.cast(confusion, dtype=tf.float32)
        confusion = confusion / tf.reduce_sum(confusion)

        # Shape: (nlabels)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self._nlabels)
        labels_freq = tf.reduce_mean(onehot_labels, axis=0)

        # Orientation test
        # confusion = tf.constant([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=tf.float32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.prediction_estimator_spec(logits, params, config)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.training_estimator_spec(loss, confusion, labels_freq, params, config)

        else:
            self.pr_curve_metric(probabilities, labels)
            return self.evaluation_estimator_spec(loss, confusion, labels_freq, params, config)
