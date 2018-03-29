import numpy as np
import tensorflow as tf

from .base_model import BaseModel
from .. import layers

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

    @property
    def generator_layers(self):
        return self._generator.layers

    @property
    def critic_layers(self):
        return self._critic.layers

    def features_placeholder(self, batch_size=1):
        return {
            'input': tf.placeholder(dtype=tf.float32,
                                    shape=(batch_size,) + self._shape,
                                    name='input')
        }

    def training_estimator_spec(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        step = tf.train.get_global_step()

        # Manually apply gradients. We want the gradients for summaries. We need
        # to apply them manually in order to avoid having duplicate gradient ops.
        gradients = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(gradients, global_step=step)

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

    def evaluation_estimator_spec(self, loss, synthetic):
        eval_metric_ops = {}

        evaluation_hooks = []
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                          loss=loss,
                                          evaluation_hooks=evaluation_hooks,
                                          eval_metric_ops=eval_metric_ops)

    def prediction_estimator_spec(self, synthetic):
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

    def model_fn(self, features, labels, mode):
        image = features['image']
        batch_size = tf.shape(image)[0]
        training=(mode==tf.estimator.ModeKeys.TRAIN)

        # Generator
        self._generator = layers.Segment(layers=self.generative_network())

        code = tf.random_uniform(shape=(batch_size,) + self._code_shape,
                                 minval=-1., maxval=1., dtype=tf.float32)

        synthetic = tf.nn.sigmoid(self._generator.apply(code, training=training))
        synthetic_ng = tf.stop_gradient(synthetic)
        tf.summary.image('gan/synthetic', tf.expand_dims(synthetic[0,...], 0))

        # Critic
        self._critic = layers.Segment(layers=self.critic_network())

        f_synth = self._critic.apply(synthetic, training=training)
        f_synth_ng = self._critic.apply(synthetic_ng, training=training)
        f_data = self._critic.apply(image, training=training)
        f_grad = tf.gradients(f_synth_ng, synthetic_ng, stop_gradients=[synthetic_ng])

        # Autoencoder
        if self._autoencoder:
            self._encoder = layers.Segment(layers=self.encoder_network())

            code_ae = self._encoder.apply(synthetic, training=training)

            tf.summary.image('ae/input', tf.expand_dims(image[0,...], 0))
            tf.summary.image('ae/synthetic', tf.expand_dims(synthetic[0,...], 0))

        # f_synth_ae = self._critic.apply(synthetic_ae, training=training)
        # f_synth_ng = self._critic.apply(tf.stop_gradient(synthetic_gen), training=training)

        # Losses
        loss_wgan = tf.reduce_mean(f_data - f_synth)

        loss_ae = tf.constant(0, dtype=tf.float32)
        if self._autoencoder:
            loss_ae = tf.nn.l2_loss(code - code_ae) / tf.cast(batch_size, dtype=tf.float32)

        loss_crit = -tf.reduce_mean(f_data - f_synth_ng)

        loss_grad = tf.square(1 - tf.nn.l2_loss(f_grad) / tf.cast(batch_size, dtype=tf.float32))

        loss = 0.01 * loss_wgan + loss_crit + 10 * loss_grad

        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.prediction_estimator_spec(synthetic)

        # Calculate Loss (for both TRAIN and EVAL modes)
        tf.summary.scalar('loss/wgan', loss_wgan)
        tf.summary.scalar('loss/grad', loss_grad)
        tf.summary.scalar('loss/ae', loss_ae)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.training_estimator_spec(loss)

        else:
            return self.evaluation_estimator_spec(loss, synthetic)
