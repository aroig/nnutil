import numpy as np
import tensorflow as tf


class ClassificationModel:
    def __init__(self, name, shape, labels):
        self._name = name
        self._shape = shape
        self._labels = labels

        self._nlabels = len(self._labels)
        self._layers = None

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
        return self._layers

    @property
    def layer_sizes(self):
        sizes = []
        for l in self._layers:
            for v in l.variables:
                sizes.append(int(np.prod(v.shape)))

        return sizes

    def features_placeholder(self, batch_size=1):
        return {
            'input': tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size,) + self._shape,
                                    name='input')
        }

    def training_estimator_spec(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        # Manually apply gradients. We want the gradients for summaries. We need
        # to apply them manually in order to avoid having duplicate gradient ops.
        gradients = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gradients, global_step=tf.train.get_global_step())

        # Gradients and weight summaries
        for grad, weight in gradients:
            # Compute per-batch normalized norms
            axis = [i for i in range(1, len(grad.shape))]
            grad_norm = tf.sqrt(tf.reduce_mean(tf.square(grad), axis=axis))
            weight_norm = tf.sqrt(tf.reduce_mean(tf.square(weight), axis=axis))

            name = weight.name.replace(':', '_')
            tf.summary.histogram('{}/weight'.format(name) , weight_norm)
            tf.summary.histogram('{}/grad'.format(name) , grad_norm)

        training_hooks = []

        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                          loss=loss,
                                          training_hooks=training_hooks,
                                          train_op=train_op)

    def evaluation_estimator_spec(self, loss, logits, labels):
        predicted_class = tf.argmax(input=logits, axis=1)

        # Shape: (nlabels, nlabels)
        confusion = tf.confusion_matrix(labels, predicted_class, self._nlabels)
        confusion = tf.cast(confusion, dtype=tf.float32)
        confusion = confusion / tf.reduce_sum(confusion)

        # Orientation test
        # confusion = tf.constant([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=tf.float32)

        # Shape: ()
        accuracy = tf.trace(confusion)

        # Shape: (nlabels)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self._nlabels)
        label_stats = tf.reduce_mean(onehot_labels, axis=0)

        # Shape: (nlabels)
        class_accuracy = tf.stack([confusion[i,i] / tf.reduce_sum(confusion[i]) for i in range(0, self._nlabels)])

        tf.summary.image('confusion', tf.expand_dims(tf.expand_dims(confusion, axis=-3), axis=-1))
        tf.summary.scalar('accuracy', accuracy)
        # tf.summary.histogram('label_stats', label_stats)

        eval_metric_ops = {
            "accuracy": tf.metrics.mean(accuracy),
            "confusion": tf.metrics.mean_tensor(confusion),
        }

        for i, lb in enumerate(self._labels):
            name = 'class_freq/{}'.format(lb)
            eval_metric_ops[name] = tf.metrics.mean(label_stats[i])

        for i, lb in enumerate(self._labels):
            name = 'class_accuracy/{}'.format(lb)
            eval_metric_ops[name] = tf.metrics.mean(class_accuracy[i])

        evaluation_hooks = []
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                          loss=loss,
                                          evaluation_hooks=evaluation_hooks,
                                          eval_metric_ops=eval_metric_ops)

    def prediction_estimator_spec(self, logits):
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

        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                          predictions=predictions,
                                          export_outputs=exports)

    def classifier_network(self):
        raise NotImplementedError

    def compose_layers(self, layers, x, training):
        for l in layers:
            if type(l) == tf.layers.Dropout:
                x = l.apply(x, training=training)
            else:
                x = l.apply(x)
        return x

    def model_fn(self, features, labels, mode):
        """Model function for classifier."""
        self._layers = self.classifier_network()

        logits = self.compose_layers(self._layers, features['image'],
                                     training=(mode==tf.estimator.ModeKeys.TRAIN))

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.prediction_estimator_spec(logits)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int32),
                                                      logits=logits)
        tf.summary.scalar('loss', loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.training_estimator_spec(loss)

        else:
            return self.evaluation_estimator_spec(loss, logits, labels)
