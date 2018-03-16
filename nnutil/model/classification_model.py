import numpy as np
import tensorflow as tf


class ClassificationModel:
    def __init__(self, name, shape, labels):
        self._name = name
        # shape: (height, width, channels)
        self._shape = shape
        self._labels = labels

        self._height = self._shape[0]
        self._width = self._shape[1]
        self._nchannels = self._shape[2]
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
                print(v)
                sizes.append(int(np.prod(v.shape)))

        return sizes

    def features_placeholder(self, batch_size=1):
        return { 'image': tf.placeholder(dtype=tf.float32, shape=(batch_size,) + self._shape, name='image') }

    def model_network(self, x, training):
        raise NotImplementedError

    def model_fn(self, features, labels, mode):
        """Model function for CNN."""
        logits = self.model_network(x=features['image'],
                                    training=(mode == tf.estimator.ModeKeys.TRAIN))

        predicted_class = tf.argmax(input=logits, axis=1)
        probabilities = tf.reshape(tf.nn.softmax(logits), shape=(-1, self._nlabels), name="probabilities")

        predictions = {
            "class": predicted_class,
            "probs": probabilities
        }

        exports = {
           'class': tf.estimator.export.ClassificationOutput(scores=probabilities)
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              export_outputs=exports)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int32), logits=logits)
        tf.summary.scalar('loss', loss)

        # Add evaluation metrics (for EVAL mode)

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

        for i, lb in enumerate(self._labels):
            name = 'class_freq/{}'.format(lb)
            tf.summary.scalar(name, label_stats[i])

        for i, lb in enumerate(self._labels):
            name = 'class_accuracy/{}'.format(lb)
            tf.summary.scalar(name, class_accuracy[i])

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
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

            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              training_hooks=training_hooks,
                                              train_op=train_op)

        else:
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
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              evaluation_hooks=evaluation_hooks,
                                              eval_metric_ops=eval_metric_ops)
