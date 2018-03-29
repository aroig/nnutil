import os
import json
import socket
import argparse

import numpy as np
import tensorflow as tf
import nnutil as nn

from .. import dataset

class Trainer:
    def __init__(self, model, dataset, build_path, cluster=None, argv=[]):
        tf.logging.set_verbosity(tf.logging.INFO)

        if cluster is not None:
            task_index = None
            hostname = socket.gethostname()

            for job, hosts in cluster.items():
                for i, h in enumerate(hosts):
                    if hostname == h.split('.')[0]:
                        task_index = i
                        job_name = job

            os.environ['TF_CONFIG'] = json.dumps({
                'cluster': cluster,
                'task': {'type': job_name, 'index': task_index}
            })

        self._build_path = os.path.abspath(build_path)

        # Launch tensorboard on the chief
        if cluster is None or job_name == 'chief':
            self._tensorboard = nn.visual.Tensorboard(self._build_path)

        self._batch_size = 32
        self._eval_fraction = 0.1

        self._model = model

        self._dataset = None
        if dataset is not None:
            self._dataset = dataset.repeat().shuffle(buffer_size=1000)


    def train(self, steps=None):
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
                                         label_key="label",
                                         resume=False)

        # Train the model
        if steps is None:
            steps = 1000

        experiment.train_and_evaluate(steps=steps)

        # Export a frozen model
        experiment.export()
        # experiment.neurallabs_export()
