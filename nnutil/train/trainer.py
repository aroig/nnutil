import os
import json
import socket
import argparse
import importlib.machinery

import numpy as np
import tensorflow as tf

from .experiment import Experiment
from .cluster import Cluster
from .. import visual
from .. import dataset
from .. import util

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

        self._cluster = Cluster(module.cluster)
        self._build_path = os.path.abspath(args.build)

        # Launch tensorboard on the chief
        self._tensorboard = None

        self._eval_fraction = 0.1

        self._model = model

        self._dataset = None
        if dataset is not None:
            self._dataset = dataset.repeat().shuffle(buffer_size=200)

    def __enter__(self):
        self._cluster.set_environment()

        if self._cluster.is_chief():
            self._tensorboard = visual.Tensorboard(self._build_path)
            self._tensorboard.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self._cluster.reset_environment()
        self._tensorboard.__exit__(type, value, traceback)


    def train(self, batch_size=32, steps=None):
        # Split up train and eval
        eval_dataset, train_dataset = dataset.partition(
            self._dataset,
            dist=[self._eval_fraction, 1 - self._eval_fraction],
            split_field='path')

        # Mutate the training set
        train_dataset = dataset.mutate_image(
            train_dataset,
            brightness=.3,
            contrast=[0.7, 1.4],
            saturation=[0.7, 1.4]
        )

        # Take batches
        eval_dataset = eval_dataset.batch(batch_size)
        train_dataset = train_dataset.batch(batch_size)

        # Visualization
        # visual.plot_sample(train_dataset)

        # Benchmark dataset pipeline
        util.benchmark_dataset(train_dataset)

        # Instance of a model attached to a dataset
        experiment = Experiment(self._build_path,
                                self._model,
                                eval_dataset=eval_dataset,
                                train_dataset=train_dataset,
                                resume=False)

        # Train the model
        if steps is None:
            steps = 1000

        experiment.train_and_evaluate(steps=steps, profiling=True)

        # Export a frozen model
        experiment.export()
        # experiment.neurallabs_export()
