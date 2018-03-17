import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl


class TestAvgPoolingModel(nl.model.NLConvNetModel):
    def __init__(self):
        nlabels = 2
        super().__init__(
            name="avg_pooling",
            shape=(2, 2, nlabels),
            labels=['{}'.format(i) for i in range(0, nlabels)])

    def model_layers(self):
        return [
            tf.layers.AveragePooling2D(pool_size=[2, 2], strides=2, name='pool0'),
            tf.layers.Flatten(),
            tf.layers.Dense(units=len(self.labels), activation=None, name='output')
        ]

class TestConv2DModel(nl.model.NLConvNetModel):
    def __init__(self):
        nlabels = 1
        super().__init__(
            name="conv2d",
            shape=(3, 3, nlabels),
            labels=['{}'.format(i) for i in range(0, nlabels)])

    def model_layers(self):
        return [
            tf.layers.Conv2D(filters=1, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu, name='conv0'),
            tf.layers.Flatten(),
            tf.layers.Dense(units=len(self.labels), activation=None, name='output')
        ]

class TestDenseModel(nl.model.NLConvNetModel):
    def __init__(self):
        nlabels = 2
        super().__init__(
            name="dense",
            shape=(1, 1, nlabels),
            labels=['{}'.format(i) for i in range(0, nlabels)])

    def model_layers(self):
        return [
            tf.layers.Flatten(),
            tf.layers.Dense(units=len(self.labels), activation=tf.nn.relu, name='dense0'),
            tf.layers.Dense(units=len(self.labels), activation=None, name='output')
        ]

class TestDropoutModel(nl.model.NLConvNetModel):
    def __init__(self):
        nlabels = 10
        super().__init__(
            name="dropout",
            shape=(1, 1, nlabels),
            labels=['{}'.format(i) for i in range(0, nlabels)])

    def model_layers(self):
        return [
            tf.layers.Flatten(),
            tf.layers.Dropout(rate=0.5),
            tf.layers.Dense(units=len(self.labels), activation=None, name='output')
        ]

class TestMaxPoolingModel(nl.model.NLConvNetModel):
    def __init__(self):
        nlabels = 2
        super().__init__(
            name="max_pooling",
            shape=(2, 2, nlabels),
            labels=['{}'.format(i) for i in range(0, nlabels)])

    def model_layers(self):
        return [
            tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2, name='pool0'),
            tf.layers.Flatten(),
            tf.layers.Dense(units=len(self.labels), activation=None, name='output')
        ]

class TestOutputModel(nl.model.NLConvNetModel):
    def __init__(self):
        nlabels=2
        super().__init__(
            name="output",
            shape=(1, 1, nlabels),
            labels=['{}'.format(i) for i in range(0, nlabels)])

    def model_layers(self):
        return [
            tf.layers.Flatten(),
            tf.layers.Dense(units=len(self.labels), activation=None, name='output')
        ]


class LayerSerialization(unittest.TestCase):
    def setUp(self):
        # Do not hide long diffs
        self.maxDiff = None

        self._path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".")
        self._experiment_path = os.path.join(self._path, "tmp")
        self._export_path = os.path.join(self._path, "tmp/export")
        self._model_path = os.path.join(self._path, "model")

    def assertFilesEqual(self, fileA, fileB):
        with open(fileA, 'r') as A_fd, open(fileB, 'r') as B_fd:
            self.assertEqual(A_fd.read(), B_fd.read())

    def assert_tf_export(self, experiment):
        model = experiment.model
        experiment.export(self._export_path, as_text=True)

        self.assertFilesEqual(os.path.join(self._model_path, "{}.pbtxt".format(model.name)),
                              os.path.join(self._export_path, "{}.pbtxt".format(model.name)))

    def assert_nl_export(self, experiment):
        model = experiment.model
        experiment.neurallabs_export(self._export_path)

        self.assertFilesEqual(os.path.join(self._model_path, "{}.net".format(model.name)),
                              os.path.join(self._export_path, "{}.net".format(model.name)))

        self.assertFilesEqual(os.path.join(self._model_path, "{}.wgt".format(model.name)),
                              os.path.join(self._export_path, "{}.wgt".format(model.name)))

    def test_serialization_output(self):
        model = TestOutputModel()

        ds = nl.dataset.labelled({ '0': nl.dataset.ones(model.shape) }, label_key="label").batch(1)

        experiment = nl.train.Experiment(self._experiment_path,
                                         model,
                                         eval_dataset=ds,
                                         train_dataset=ds,
                                         seed=42)

        # TODO: generate checkpoint without training step
        experiment.train(steps=1)
        self.assert_tf_export(experiment)
        self.assert_nl_export(experiment)

    def test_serialization_dense(self):
        model = TestDenseModel()

        ds = nl.dataset.labelled({ '0': nl.dataset.ones(model.shape) }, label_key="label").batch(1)

        experiment = nl.train.Experiment(self._experiment_path,
                                         model,
                                         eval_dataset=ds,
                                         train_dataset=ds,
                                         seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)
        self.assert_nl_export(experiment)

    def test_serialization_avg_pooling(self):
        model = TestAvgPoolingModel()

        ds = nl.dataset.labelled({ '0': nl.dataset.ones(model.shape) }, label_key="label").batch(1)

        experiment = nl.train.Experiment(self._experiment_path,
                                         model,
                                         eval_dataset=ds,
                                         train_dataset=ds,
                                         seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)
        self.assert_nl_export(experiment)

    def test_serialization_max_pooling(self):
        model = TestMaxPoolingModel()

        ds = nl.dataset.labelled({ '0': nl.dataset.ones(model.shape) }, label_key="label").batch(1)

        experiment = nl.train.Experiment(self._experiment_path,
                                         model,
                                         eval_dataset=ds,
                                         train_dataset=ds,
                                         seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)
        self.assert_nl_export(experiment)

    def test_serialization_conv2d(self):
        model = TestConv2DModel()

        ds = nl.dataset.labelled({ '0': nl.dataset.ones(model.shape) }, label_key="label").batch(1)

        experiment = nl.train.Experiment(self._experiment_path,
                                         model,
                                         eval_dataset=ds,
                                         train_dataset=ds,
                                         seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)
        self.assert_nl_export(experiment)

    def test_serialization_dropout(self):
        model = TestDropoutModel()

        ds = nl.dataset.labelled({ '0': nl.dataset.ones(model.shape) }, label_key="label").batch(1)

        experiment = nl.train.Experiment(self._experiment_path,
                                         model,
                                         eval_dataset=ds,
                                         train_dataset=ds,
                                         seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)
        self.assert_nl_export(experiment)


if __name__ == '__main__':
    unittest.main()
