import os
import subprocess
from datetime import datetime

import tensorflow as tf
import numpy as np

from tensorflow.python import debug as tfdbg

from .. import visual
from .. import model

from .tensorboard_profiler_hook import TensorboardProfilerHook

class Experiment:
    def __init__(self, path, model, eval_dataset_fn=None, train_dataset_fn=None,
                 hyperparameters=None, resume=False, debug=False, seed=None,
                 eval_secs=None, log_secs=None, profile_secs=None, name=None):

        default_hyperparameters = {
            'batch_size': 64,
            'learning_rate': 0.001,
            'learning_rate_decay': 1.0,
            'momentum': 0.9,
            'train_steps': 1000,
            'eval_steps': 10,
            'regularizer': 0.0,
            'regularizer_step': 1000,
            'sample_bias': 0.0,
            'sample_bias_step': 1000
        }

        if hyperparameters is None:
            hyperparameters = {}

        self._hyperparameters = {}
        for k, default in default_hyperparameters.items():
            self._hyperparameters[k] = hyperparameters.get(k, default)

        self._model = model

        self._train_dataset_fn = train_dataset_fn
        self._eval_dataset_fn = eval_dataset_fn

        self._seed = seed

        self._resume = resume
        self._debug = debug

        if eval_secs is None:
            eval_secs = 120

        if log_secs is None:
            log_secs = 120

        self._profile_secs = profile_secs
        self._log_secs = log_secs
        self._checkpoint_secs = log_secs
        self._eval_secs = eval_secs
        self._summary_steps = 30

        if name is None:
            # TODO: mix model and dataset names
            name = self._model.name
        self._name = name

        # Path to the model directory
        self._path = None
        if path is not None:
            self._path = os.path.abspath(path)

        self._run_timestamp = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        if self._resume:
            all_runs = []

            train_dir = self._path
            if os.path.exists(train_dir):
                all_runs = list(sorted(os.listdir(train_dir)))

            if len(all_runs) > 0:
                self._run_timestamp = all_runs[-1]

        if self._path is not None and not os.path.exists(self.path):
            os.makedirs(self.path)

    @property
    def path(self):
        return os.path.join(self._path, self._run_timestamp)

    @property
    def model(self):
        return self._model

    def hooks(self, mode):
        hooks = []

        if self._debug:
            hooks.append(
                tfdbg.TensorBoardDebugHook("localhost:6007")
            )

        if self._profile_secs is not None:
            hooks.append(
                TensorboardProfilerHook(
                    save_secs=self._log_secs,
                    output_dir=self.path)
            )

        return hooks

    def estimator(self, mode):
        model_fn = self._model.model_fn

        config = tf.estimator.RunConfig(
            model_dir=self.path,
            tf_random_seed=self._seed,
            save_summary_steps=self._summary_steps,
            save_checkpoints_secs=self._checkpoint_secs,
            keep_checkpoint_max=10,
            log_step_count_steps=self._summary_steps)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=self.path,
            config=config,
            params=self._hyperparameters)

        return estimator

    def train(self):
        train_steps = self._hyperparameters.get('train_steps', 1000)
        batch_size = self._hyperparameters.get('batch_size', 64)

        estimator = self.estimator('train')
        hooks = self.hooks('train')

        estimator.train(
            input_fn=lambda: self._train_dataset_fn().batch(batch_size).prefetch(1),
            steps=train_steps,
            hooks=hooks)

    def evaluate(self):
        eval_steps = self._hyperparameters.get('eval_steps', 10)
        batch_size = self._hyperparameters.get('batch_size', 64)

        estimator = self.estimator('eval')
        hooks = self.hooks('eval')

        results = estimator.evaluate(
            input_fn=lambda: self._eval_dataset_fn().batch(batch_size).prefetch(1),
            steps=eval_steps,
            hooks=hooks)

        return results

    def train_and_evaluate(self):
        train_steps = self._hyperparameters.get('train_steps', 1000)
        eval_steps = self._hyperparameters.get('eval_steps', 10)
        batch_size = self._hyperparameters.get('batch_size', 64)

        estimator = self.estimator('train')
        train_hooks = self.hooks('train')
        evaluation_hooks = self.hooks('eval')

        tf.estimator.train_and_evaluate(
            estimator,
            tf.estimator.TrainSpec(
                hooks=train_hooks,
                input_fn=lambda: self._train_dataset_fn().batch(batch_size).prefetch(1),
                max_steps=train_steps),
            tf.estimator.EvalSpec(
                hooks=evaluation_hooks,
                input_fn=lambda: self._eval_dataset_fn().batch(batch_size).prefetch(1),
                throttle_secs=self._eval_secs,
                steps=eval_steps))

    def export(self, **kwargs):
        return self.plain_export(**kwargs)

    def serving_export(self, export_path=None, as_text=False):
        if export_path is None:
            export_path = os.path.join(self.path, "export")

        model = self._model
        def input_receiver_fn():
            # TODO: we should receive a serialized input and parse it
            # serialized_example = tf.placeholder(dtype=tf.string, shape=[None], name='serialized_input')
            # receiver = {'inputs':  serialized_example}
            # features = {'blah': tf.parse_example(serialized_example, feature_spec)}
            receiver = {}
            features = model.features_placeholder()
            return tf.estimator.export.ServingInputReceiver(features, receiver)

        if not os.path.isdir(export_path):
            os.makedirs(export_path)

        # TODO: need a 'predict' tag?
        estimator = self.estimator('train')

        savemodel_path = estimator.export_savedmodel(
            export_path,
            input_receiver_fn,
            as_text=True)

        savemodel_path = savemodel_path.decode()

        return savemodel_path

    def frozen_serving_export(self, export_path=None, batch_size=1, as_text=False):
        if export_path is None:
            export_path = os.path.join(self.path, "export")

        savemodel_path = self.serving_export(export_path=export_path, as_text=as_text)

        # Obtain list of output ops
        with tf.Session(graph=tf.Graph()) as sess:
            features = self._model.features_placeholder()

            estimator_spec = self._model.model_fn(
                features,
                {},
                tf.estimator.ModeKeys.PREDICT,
                {},
                {}
            )

            output_ops = [x.op.name for k, x in estimator_spec.predictions.items()]

        # Freeze exported graph
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], savemodel_path)

            # NOTE: add_shapes adds `_output_shapes` members
            graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

            # Transform variables to constants and strip unused ops
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                output_ops
            )

            if as_text:
                model_pb = '{}.pbtxt'.format(self._name)
            else:
                model_pb = '{}.pb'.format(self._name)

            tf.train.write_graph(
                frozen_graph_def,
                export_path,
                model_pb,
                as_text=as_text)

    def plain_export(self, export_path=None, batch_size=1, as_text=False):
        if export_path is None:
            export_path = os.path.join(self.path, "export")

        checkpoint_path = tf.train.latest_checkpoint(self.path)

        with tf.Session(graph=tf.Graph()) as sess:
            features = self._model.features_placeholder()

            estimator_spec = self._model.model_fn(
                features,
                {},
                tf.estimator.ModeKeys.PREDICT,
                {},
                {})

            output_ops = [x.op.name for k, x in estimator_spec.predictions.items()]

            saver = tf.train.Saver()

            # Restore variables from checkpoint
            # sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint_path)

            # NOTE: add_shapes adds `_output_shapes` members
            graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

            frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                output_ops
            )

            # Export unfrozen graph
            tf.train.write_graph(
                graph_def,
                export_path,
                '{}.pbtxt'.format(self._name),
                as_text=True)

            # Export frozen graph
            tf.train.write_graph(
                frozen_graph_def,
                export_path,
                '{}.pb'.format(self._name),
                as_text=as_text)
