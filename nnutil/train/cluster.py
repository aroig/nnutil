import socket
import os
import json

import tensorflow as tf

from .. import visual

class Cluster:
    def __init__(self, cluster_spec, tensorboard_path=None):
        self._cluster = cluster_spec

        self._tensorboard_path = tensorboard_path
        self._tensorboard = None

    def current_task(self):
        hostname = socket.gethostname()
        for job, hosts in self._cluster.items():
            for i, h in enumerate(hosts):
                if hostname == h.split('.')[0]:
                    task_index = i
                    job_name = job

        return job_name, task_index

    def set_environment(self):
        if self._cluster is not None:
            job_name, task_index = self.current_task()

            os.environ['TF_CONFIG'] = json.dumps({
                'cluster': self._cluster,
                'task': {'type': job_name, 'index': task_index}
            })

    def reset_environment(self):
        if self._cluster is not None:
            del os.environ['TF_CONFIG']

    def start_server(self):
        if self._cluster is not None:
            job_name, task_index = self.current_task()
            cluster_spec = tf.train.ClusterSpec(self._cluster)
            server = tf.train.Server(cluster_spec, job_name=job_name, task_index=task_index)
            server.start()
            server.join()

    def is_chief(self):
        if self._cluster is not None:
            job_name, task_index = self.current_task()
            return (job_name == 'chief')
        else:
            return True

    def __enter__(self):
        self.set_environment()

        if self.is_chief() and self._tensorboard_path is not None:
            self._tensorboard = visual.Tensorboard(self._tensorboard_path)
            self._tensorboard.__enter__()

        return self

    def __exit__(self, type, value, traceback):
        self.reset_environment()
        if self._tensorboard is not None:
            self._tensorboard.__exit__(type, value, traceback)
