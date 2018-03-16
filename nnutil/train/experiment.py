import os
import subprocess
from datetime import datetime

import tensorflow as tf
import numpy as np

from .tensorboard_profiler_hook import TensorboardProfilerHook
from ..model import NLModelWriter

class Experiment:
    def __init__(self, path, model, eval_dataset=None, train_dataset=None, label_key="label", seed=None):
        self._model = model
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._label_key = label_key
        self._seed = seed

        self._log_secs = 60
        self._checkpoint_secs = 60
        self._summary_steps = 10

        # TODO: make up a name mixing model and dataset names
        self._name = self._model.name

        # Path to the model directory
        self._path = os.path.abspath(path)

        # Cache the run timestamp, so it does not change in a single run
        self._run_timestamp = None

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    @property
    def path(self):
        name = self._model.name
        return os.path.join(self._path, name)

    @property
    def model(self):
        return self._model

    def get_run_path(self, mode, resume=False):
        path = self.path

        if resume and self._run_timestamp is None:
            all_runs = []

            train_dir = os.path.join(path, 'train')
            if os.path.exists(train_dir):
                all_runs = list(sorted(os.listdir(train_dir)))

            if len(all_runs) > 0:
                self._run_timestamp = all_runs[-1]

        if self._run_timestamp is None:
            self._run_timestamp = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        run_path = os.path.join(path, mode, self._run_timestamp)

        if not os.path.exists(run_path):
            os.makedirs(run_path)

        return run_path


    def hooks(self, mode):
        hooks = []

        log_tensors = {
            # "loss": "loss",
            # "confusion": "confusion"
            # "probabilities": "probabilities"
        }

        hooks.append(
            tf.train.LoggingTensorHook(
                every_n_secs=self._log_secs,
                tensors=log_tensors))

        if mode == 'prof':
            path = self.get_run_path('prof', resume=False)
            hooks.append(TensorboardProfilerHook(save_secs=self._log_secs, output_dir=path))

        return hooks


    def estimator(self, mode, resume=False):
        path = self.get_run_path(mode, resume=resume)
        model_fn = self._model.model_fn

        config = tf.estimator.RunConfig(
            model_dir=path,
            tf_random_seed=self._seed,
            save_summary_steps=self._summary_steps,
            save_checkpoints_secs=self._checkpoint_secs,
            keep_checkpoint_max=10,
            log_step_count_steps=self._summary_steps)

        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=path, config=config)

        return estimator

    def iterator(self, ds):
        if self._label_key is not None:
            ds = ds.map(lambda x: (x, x[self._label_key]))

        return ds.make_one_shot_iterator().get_next()

    def profile(self, steps=200):
        train_dataset = self._train_dataset
        def input_fn():
           return  self.iterator(train_dataset)

        hooks = self.hooks('prof')

        estimator = self.estimator('prof', resume=False)
        estimator.train(input_fn=input_fn, steps=steps, hooks=hooks)


    def train(self, steps=2000, resume=False):
        train_dataset = self._train_dataset
        def input_fn():
           return self.iterator(train_dataset)

        hooks = self.hooks('train')

        estimator = self.estimator('train', resume=resume)
        estimator.train(input_fn=input_fn, steps=steps, hooks=hooks)


    def evaluate(self):
        eval_dataset = self._eval_dataset
        def input_fn():
            return self.iterator(eval_dataset)

        hooks = self.hooks('eval')

        estimator = self.estimator('eval', resume=False)
        results = estimator.evaluate(input_fn=input_fn, hooks=hooks)
        return results


    def train_and_evaluate(self, steps=2000, resume=False):

        train_dataset = self._train_dataset
        def train_input_fn():
            return self.iterator(train_dataset)

        eval_dataset = self._eval_dataset
        def eval_input_fn():
            return self.iterator(eval_dataset)

        train_hooks = self.hooks('train')
        evaluation_hooks = []

        estimator = self.estimator('train', resume=resume)
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
                steps=5))


    def export(self, export_path, batch_size=1, as_text=False):
        model = self._model
        def input_receiver_fn():
            features = model.features_placeholder()
            receiver = model.features_placeholder()
            return tf.estimator.export.ServingInputReceiver(features, receiver)

        if not os.path.isdir(export_path):
            os.makedirs(export_path)

        # TODO: need a 'predict' tag?
        estimator = self.estimator('train', resume=True)
        savemodel_path = estimator.export_savedmodel(export_path, input_receiver_fn, as_text=True)
        savemodel_path = savemodel_path.decode()

        # Freeze exported graph
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [ tf.saved_model.tag_constants.SERVING ], savemodel_path)

            # Transform variables to constants and strip unused ops
            frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                [ "probabilities" ]
            )

            if as_text:
                model_pb = '{}.pbtxt'.format(self._name)
            else:
                model_pb = '{}.pb'.format(self._name)

            tf.train.write_graph(frozen_graph_def, export_path, model_pb, as_text=as_text)


    def neurallabs_export(self, export_path):
        model = self._model
        estimator = self.estimator('train', resume=True)

        # TODO: Find a better way to extract the weights. We need to setup an estimator within a session of our own.
        # This is an adaptation of tf.estimator.Estimator.predict
        checkpoint_path = tf.train.latest_checkpoint(estimator._model_dir)
        if not checkpoint_path:
            raise ValueError('Could not find trained model in model_dir: {}.'.format(estimator._model_dir))

        eval_dataset = self._eval_dataset
        def input_fn():
            return self.iterator(eval_dataset)

        with tf.Graph().as_default() as g:
            tf.set_random_seed(estimator._config.tf_random_seed)
            estimator._create_and_assert_global_step(g)

            features, input_hooks = estimator._get_features_from_input_fn(input_fn, tf.estimator.ModeKeys.PREDICT)
            estimator_spec = estimator._call_model_fn(features, None, tf.estimator.ModeKeys.PREDICT, estimator.config)
            predictions = estimator._extract_keys(estimator_spec.predictions, None)

            with tf.train.MonitoredSession(
                session_creator = tf.train.ChiefSessionCreator(
                    checkpoint_filename_with_path = checkpoint_path,
                    scaffold = estimator_spec.scaffold,
                    config = estimator._session_config),
                hooks = []) as sess:

                writer = NLModelWriter(model, sess)

                net_path = os.path.join(export_path, "{}.net".format(self._name))
                wgt_path = os.path.join(export_path, "{}.wgt".format(self._name))

                with open(net_path, 'w') as net_fd, open(wgt_path, 'w') as wgt_fd:
                    writer.write(net_fd, wgt_fd)


    def plain_export(self, export_path, batch_size = 1):
        with tf.Session(graph=tf.Graph()) as sess:
            features = self._model.features_placeholder()
            probs = self._model.model_fn(features, {}, tf.estimator.ModeKeys.PREDICT).predictions['probs']

            sess.run(tf.global_variables_initializer())
            # TODO: need to load variables from the last checkpoint

            frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                [ probs.op.name ]
            )

            tf.train.write_graph(sess.graph, export_path, '{}.pbtxt'.format(self._name), as_text=True)
            tf.train.write_graph(frozen_graph_def, export_path, '{}.pb'.format(self._name), as_text=False)
