import os
import json
import socket
import argparse
import importlib.machinery

import numpy as np
import tensorflow as tf

import nnutil as nn

class Trainer:
    def __init__(self, model, dataset, argv=[]):
        tf.logging.set_verbosity(tf.logging.INFO)

        parser = argparse.ArgumentParser()
        parser.add_argument('--build', help='Dataset path')
        parser.add_argument('--cluster', help='Cluster spec path', default=None)
        args = parser.parse_args(argv)

        # Load cluster_spec
        loader = importlib.machinery.SourceFileLoader("cluster", args.cluster)
        module = loader.load_module("cluster")

        self._cluster = nn.train.Cluster(module.cluster)
        self._build_path = os.path.abspath(args.build)

        # Launch tensorboard on the chief
        self._tensorboard = None

        self._eval_fraction = 0.1

        self._model = model

        self._dataset = None
        if dataset is not None:
            self._dataset = dataset.repeat().shuffle(buffer_size=1000)

    def __enter__(self):
        self._cluster.set_environment()

        if self._cluster.is_chief():
            self._tensorboard = nn.visual.Tensorboard(self._build_path)
            self._tensorboard.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self._cluster.reset_environment()
        self._tensorboard.__exit__(type, value, traceback)


    def train(self, batch_size=32, steps=None):
        # Split up train and eval
        eval_dataset, train_dataset = nn.dataset.partition(
            self._dataset,
            dist=[self._eval_fraction, 1 - self._eval_fraction],
            key_fn=lambda x: x[0]['path'])

        # Mutate the training set
        train_dataset = nn.dataset.mutate(train_dataset)

        # Take batches
        eval_dataset = eval_dataset.batch(batch_size)
        train_dataset = train_dataset.batch(batch_size)

        # Visualization
        # nl.visual.plot_sample(train_dataset)

        # Instance of a model attached to a dataset
        experiment = nn.train.Experiment(self._build_path,
                                         self._model,
                                         eval_dataset=eval_dataset,
                                         train_dataset=train_dataset,
                                         label_key="label",
                                         resume=False)

        # Train the model
        if steps is None:
            steps = 1000

        experiment.train_and_evaluate(steps=steps)

        # Export a frozen model
        experiment.export()
        # experiment.neurallabs_export()
