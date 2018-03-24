import os
import argparse

import numpy as np
import tensorflow as tf
import nnutil as nn

from .. import dataset

class Trainer:
    def __init__(self, model, dataset, build_path, argv=[]):
        tf.logging.set_verbosity(tf.logging.INFO)

        self._build_path = os.path.abspath(build_path)

        self._batch_size = 32
        self._eval_fraction = 0.1

        self._model = model

        self._dataset = None
        if dataset is not None:
            self._dataset = dataset.repeat().shuffle(buffer_size=1000)


    def train(self):
        # Split up train and eval
        eval_dataset, train_dataset = dataset.partition(
            self._dataset,
            dist=[self._eval_fraction, 1 - self._eval_fraction],
            key_fn=lambda x: x[0]['path'])

        # Mutate the training set
        train_dataset = nn.dataset.mutate(train_dataset)

        # Take batches
        eval_dataset = eval_dataset.batch(self._batch_size)
        train_dataset = train_dataset.batch(self._batch_size)

        # Visualization
        # nl.visual.plot_sample(train_dataset)

        # Instance of a model attached to a dataset
        experiment = nn.train.Experiment(self._build_path,
                                         self._model,
                                         eval_dataset=eval_dataset,
                                         train_dataset=train_dataset,
                                         label_key="label")

        # Launch tensorboard
        # tb = nn.visual.Tensorboard(self._build_path)

        # Train the model
        experiment.train_and_evaluate(steps=2000, resume=False)
        # experiment.train(steps=10, resume=False)

        # Export a frozen model
        export_path = os.path.join(experiment.path, "export")
        experiment.export(export_path)
        experiment.neurallabs_export(export_path)
