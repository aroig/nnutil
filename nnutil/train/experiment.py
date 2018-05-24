import os
import subprocess
from datetime import datetime

import tensorflow as tf
import numpy as np

from .. import model
from .tensorboard_profiler_hook import TensorboardProfilerHook

class Experiment:
    def __init__(self, path, model, eval_dataset=None, train_dataset=None,
                 hyperparameters=None, resume=False, seed=None):
        if hyperparameters is None:
            hyperparameters = {}
        self._model = model
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._hyperparameters = hyperparameters
        self._seed = seed

        self._profile_secs = 120
        self._log_secs = 120
        self._checkpoint_secs = 120
        self._summary_steps = 10
        self._eval_steps = 5

        # TODO: make up a name mixing model and dataset names
        self._name = self._model.name

        # Path to the model directory
        self._path = os.path.abspath(path)

        self._run_timestamp = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        if resume:
            all_runs = []

            train_dir = self._path
            if os.path.exists(train_dir):
                all_runs = list(sorted(os.listdir(train_dir)))

            if len(all_runs) > 0:
                self._run_timestamp = all_runs[-1]

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    @property
    def path(self):
        return os.path.join(self._path, self._run_timestamp)

    @property
    def model(self):
        return self._model

    def hooks(self, mode, profiling=False):
        hooks = []

        if profiling and self._profile_secs is not None:
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

        estimator = tf.estimator.Estimator(model_fn=model_fn,
                                           model_dir=self.path,
                                           config=config,
                                           params=self._hyperparameters)

        return estimator

    def iterator(self, ds):
        return ds.make_one_shot_iterator().get_next()

    def train(self, steps=2000, profiling=False):
        train_dataset = self._train_dataset
        def input_fn():
           return self.iterator(train_dataset)

        estimator = self.estimator('train')
        hooks = self.hooks('train', profiling=profiling)

        estimator.train(input_fn=input_fn, steps=steps, hooks=hooks)

    def evaluate(self, steps, name=None):
        eval_dataset = self._eval_dataset
        def input_fn():
            return self.iterator(eval_dataset)

        estimator = self.estimator('eval')
        hooks = self.hooks('eval')

        results = estimator.evaluate(input_fn=input_fn, steps=steps, hooks=hooks, name=name)
        return results

    def train_and_evaluate(self, steps=2000, profiling=False):

        train_dataset = self._train_dataset
        def train_input_fn():
            return self.iterator(train_dataset)

        eval_dataset = self._eval_dataset
        def eval_input_fn():
            return self.iterator(eval_dataset)

        estimator = self.estimator('train')
        train_hooks = self.hooks('train')
        evaluation_hooks = self.hooks('eval', profiling=profiling)

        tf.estimator.train_and_evaluate(
            estimator,
            tf.estimator.TrainSpec(
                hooks=train_hooks,
                input_fn=train_input_fn,
                max_steps=steps),
            tf.estimator.EvalSpec(
                hooks=evaluation_hooks,
                input_fn=eval_input_fn,
                throttle_secs=self._checkpoint_secs,
                steps=self._eval_steps))

    def serving_export(self, export_path=None, as_text=False):
        if export_path is None:
            export_path = os.path.join(self.path, "export")

        model = self._model
        def input_receiver_fn():
            features = model.features_placeholder()
            receiver = model.features_placeholder()
            return tf.estimator.export.ServingInputReceiver(features, receiver)

        if not os.path.isdir(export_path):
            os.makedirs(export_path)

        # TODO: need a 'predict' tag?
        estimator = self.estimator('train')
        savemodel_path = estimator.export_savedmodel(export_path, input_receiver_fn, as_text=True)
        savemodel_path = savemodel_path.decode()

        return savemodel_path

    def export(self, export_path=None, batch_size=1, as_text=False):
        if export_path is None:
            export_path = os.path.join(self.path, "export")

        savemodel_path = self.serving_export(export_path=export_path, as_text=as_text)

        # Freeze exported graph
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], savemodel_path)

            # TODO: need to get list of output ops from the model

            # Transform variables to constants and strip unused ops
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(add_shapes=True),
                ["probabilities"]
            )

            if as_text:
                model_pb = '{}.pbtxt'.format(self._name)
            else:
                model_pb = '{}.pb'.format(self._name)

            tf.train.write_graph(frozen_graph_def, export_path, model_pb, as_text=as_text)

    def plain_export(self, export_path=None, batch_size = 1):
        if export_path is None:
            export_path = os.path.join(self.path, "export")

        with tf.Session(graph=tf.Graph()) as sess:
            features = self._model.features_placeholder()

            estimator_spec = self._model.model_fn(features,
                                               {},
                                               tf.estimator.ModeKeys.PREDICT)

            # TODO: need to get list of predictions from the model

            probs = estimator_spec.predictions['probs']

            sess.run(tf.global_variables_initializer())

            # TODO: need to load variables from the last checkpoint

            frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                [ probs.op.name ]
            )

            tf.train.write_graph(sess.graph,
                                 export_path,
                                 '{}.pbtxt'.format(self._name),
                                 as_text=True)

            tf.train.write_graph(frozen_graph_def,
                                 export_path,
                                 '{}.pb'.format(self._name),
                                 as_text=False)
