import os
import subprocess
from datetime import datetime

import tensorflow as tf
import numpy as np

from .. import model
from .tensorboard_profiler_hook import TensorboardProfilerHook

import nnutil as nn

class Experiment:
    def __init__(self, path, model, eval_dataset=None, train_dataset=None, resume=False, label_key="label", seed=None):
        self._model = model
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._label_key = label_key
        self._seed = seed

        self._log_secs = 60
        self._checkpoint_secs = 60
        self._summary_steps = 10
        self._eval_steps = 5

        # TODO: make up a name mixing model and dataset names
        self._name = self._model.name

        # Path to the model directory
        self._path = os.path.abspath(path)

        self._run_timestamp = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        if resume:
            all_runs = []

            train_dir = os.path.join(path, self._name)
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

    def hooks(self, mode):
        hooks = []

        if mode == 'prof':
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

        # hyperparameters
        params = {}

        estimator = tf.estimator.Estimator(model_fn=model_fn,
                                           model_dir=self.path,
                                           config=config,
                                           params=params)

        return estimator

    def iterator(self, ds):
        if self._label_key is not None:
            ds = ds.map(lambda x: (x, x[self._label_key]))

        return ds.make_one_shot_iterator().get_next()

    def profile(self, steps=200):
        train_dataset = self._train_dataset
        def input_fn():
           return  self.iterator(train_dataset)

        estimator = self.estimator('prof')
        hooks = self.hooks('prof')

        estimator.train(input_fn=input_fn, steps=steps, hooks=hooks)


    def train(self, steps=2000):
        train_dataset = self._train_dataset
        def input_fn():
           return self.iterator(train_dataset)

        estimator = self.estimator('train')
        hooks = self.hooks('train')

        estimator.train(input_fn=input_fn, steps=steps, hooks=hooks)


    def evaluate(self):
        eval_dataset = self._eval_dataset
        def input_fn():
            return self.iterator(eval_dataset)

        estimator = self.estimator('eval')
        hooks = self.hooks('eval')

        results = estimator.evaluate(input_fn=input_fn, hooks=hooks)
        return results


    def train_and_evaluate(self, steps=2000):

        train_dataset = self._train_dataset
        def train_input_fn():
            return self.iterator(train_dataset)

        eval_dataset = self._eval_dataset
        def eval_input_fn():
            return self.iterator(eval_dataset)

        estimator = self.estimator('train')
        train_hooks = self.hooks('train')
        evaluation_hooks = self.hooks('eval')

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


    def export(self, export_path=None, batch_size=1, as_text=False):
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


    def neurallabs_export(self, export_path=None):
        if export_path is None:
            export_path = os.path.join(self.path, "export")

        model = self._model
        estimator = self.estimator('train')

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

                writer = model.NLModelWriter(model, sess)

                net_path = os.path.join(export_path, "{}.net".format(self._name))
                wgt_path = os.path.join(export_path, "{}.wgt".format(self._name))

                with open(net_path, 'w') as net_fd, open(wgt_path, 'w') as wgt_fd:
                    writer.write(net_fd, wgt_fd)


    def plain_export(self, export_path=None, batch_size = 1):
        if export_path is None:
            export_path = os.path.join(self.path, "export")

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
