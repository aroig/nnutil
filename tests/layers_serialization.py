import unittest
import os

import numpy as np
import tensorflow as tf
import nnutil as nl

out_args = {
    'kernel_initializer': tf.initializers.identity(),
    'bias_initializer': tf.initializers.zeros(),
    'activation': None
}

class TestAvgPoolingModel(nl.model.ClassificationModel):
    def __init__(self):
        nlabels = 2
        super().__init__(
            name="avg_pooling",
            shape=(2, 2, nlabels),
            labels=['{}'.format(i) for i in range(0, nlabels)],
            outfunction=tf.identity)

    def classifier_network(self, params):
        return [
            tf.layers.AveragePooling2D(pool_size=2, strides=2, name='pool0'),
            tf.layers.Flatten(),
            tf.layers.Dense(units=len(self.labels), **out_args, name='output')
        ]

class TestConv2DModel(nl.model.ClassificationModel):
    def __init__(self, kernel_size):
        nlabels = 2
        self._kernel_size = kernel_size
        super().__init__(
            name="conv2d_{}".format(kernel_size),
            shape=(kernel_size, kernel_size, nlabels),
            labels=['{}'.format(i) for i in range(0, nlabels)],
            outfunction=tf.identity)

    def classifier_network(self, params):
        args = {
            'kernel_initializer': tf.initializers.random_normal(seed=42),
            'bias_initializer': tf.initializers.random_normal(seed=43),
            'padding': 'valid',
            'activation': None
        }
        return [
            tf.layers.Conv2D(filters=len(self.labels), kernel_size=self._kernel_size, **args, name='conv0'),
            tf.layers.Flatten(),
            tf.layers.Dense(units=len(self.labels), **out_args, name='output')
        ]

class TestSeparableConv2DModel(nl.model.ClassificationModel):
    def __init__(self, kernel_size):
        self._kernel_size = kernel_size
        nlabels = 2
        super().__init__(
            name="separable_conv2d_{}".format(kernel_size),
            shape=(kernel_size, kernel_size, nlabels),
            labels=['{}'.format(i) for i in range(0, nlabels)],
            outfunction=tf.identity)

    def classifier_network(self, params):
        args = {
            'depthwise_initializer': tf.initializers.random_normal(seed=42),
            'pointwise_initializer': tf.initializers.random_normal(seed=43),
            'bias_initializer': tf.initializers.random_normal(seed=44),
            'padding': 'valid',
            'activation': None,
            'depth_multiplier': 1
        }

        layers = [
            tf.layers.SeparableConv2D(filters=len(self.labels), kernel_size=self._kernel_size, **args, name='conv0'),
            tf.layers.Flatten(),
            tf.layers.Dense(units=len(self.labels), **out_args, name='output')
        ]

        # Workaround for bug in SeparableConv2D layer present in TF 1.7
        layers[0].bias_initializer = tf.initializers.random_normal(seed=44)

        return layers

class TestDenseModel(nl.model.ClassificationModel):
    def __init__(self):
        nlabels = 2
        super().__init__(
            name="dense",
            shape=(1, 1, nlabels),
            labels=['{}'.format(i) for i in range(0, nlabels)],
            outfunction=tf.identity)

    def classifier_network(self, params):
        args = {
            'kernel_initializer': tf.initializers.random_normal(seed=42),
            'bias_initializer': tf.initializers.random_normal(seed=43),
            'activation': None
        }
        return [
            tf.layers.Flatten(),
            tf.layers.Dense(units=len(self.labels), **args, name='dense'),
            tf.layers.Dense(units=len(self.labels), **out_args, name='output')
        ]

class TestDropoutModel(nl.model.ClassificationModel):
    def __init__(self):
        nlabels = 10
        super().__init__(
            name="dropout",
            shape=(1, 1, nlabels),
            labels=['{}'.format(i) for i in range(0, nlabels)],
            outfunction=tf.identity)

    def classifier_network(self, params):
        return [
            tf.layers.Flatten(),
            tf.layers.Dropout(rate=0.5, seed=42),
            tf.layers.Dense(units=len(self.labels), **out_args, name='output')
        ]

class TestMaxPoolingModel(nl.model.ClassificationModel):
    def __init__(self):
        nlabels = 2
        super().__init__(
            name="max_pooling",
            shape=(2, 2, nlabels),
            labels=['{}'.format(i) for i in range(0, nlabels)],
            outfunction=tf.identity)

    def classifier_network(self, params):
        return [
            tf.layers.MaxPooling2D(pool_size=2, strides=2, name='pool0'),
            tf.layers.Flatten(),
            tf.layers.Dense(units=len(self.labels), **out_args, name='output')
        ]

class TestOutputModel(nl.model.ClassificationModel):
    def __init__(self):
        nlabels=2
        super().__init__(
            name="output",
            shape=(1, 1, nlabels),
            labels=['{}'.format(i) for i in range(0, nlabels)],
            outfunction=tf.nn.softmax)

    def classifier_network(self, params):
        args = {
            'kernel_initializer': tf.initializers.random_normal(seed=42),
            'bias_initializer': tf.initializers.random_normal(seed=43),
            'activation': None
        }
        return [
            tf.layers.Flatten(),
            tf.layers.Dense(units=len(self.labels), **args, name='output')
        ]


class TestInputAModel(nl.model.ClassificationModel):
    def __init__(self):
        nlabels=4
        super().__init__(
            name="inputA",
            shape=(2, 2, 1),
            labels=['{}'.format(i) for i in range(0, nlabels)],
            outfunction=tf.identity)

    def classifier_network(self, params):
        return [
            nl.layers.LayerNormalization(axis=[0, 1]),
            tf.layers.Flatten(),
            tf.layers.Dense(units=len(self.labels), **out_args, name='output')
        ]


class TestInputBModel(nl.model.ClassificationModel):
    def __init__(self):
        nlabels=4
        super().__init__(
            name="inputB",
            shape=(2, 2, 1),
            labels=['{}'.format(i) for i in range(0, nlabels)],
            outfunction=tf.identity)

    def classifier_network(self, params):
        return [
            nl.layers.RangeNormalization(minval=1, maxval=2, axis=[0, 1]),
            tf.layers.Flatten(),
            tf.layers.Dense(units=len(self.labels), **out_args, name='output')
        ]


class LayerSerialization(unittest.TestCase):
    def setUp(self):
        # Do not hide long diffs
        self.maxDiff = None

        self._path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../Tests")
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

    def test_serialization_output(self):
        model = TestOutputModel()

        ds = nl.dataset.labelled({'0': nl.dataset.ones(model.input_shape)}, label_key="label").batch(1)

        experiment = nl.train.NLExperiment(
            self._experiment_path,
            model,
            eval_dataset=ds,
            train_dataset=ds,
            hyperparameters={'learning_rate': 0},
            seed=42)

    def test_serialization_output(self):
        model = TestOutputModel()

        ds = nl.dataset.labelled({'0': nl.dataset.ones(model.input_shape)}, label_key="label").batch(1)

        experiment = nl.train.NLExperiment(
            self._experiment_path,
            model,
            eval_dataset=ds,
            train_dataset=ds,
            hyperparameters={'learning_rate': 0},
            seed=42)

        # TODO: generate checkpoint without training step
        experiment.train(steps=1)
        self.assert_tf_export(experiment)

    def test_serialization_input_layernorm(self):
        model = TestInputAModel()

        shape = model.input_shape
        tsr_ds = tf.data.Dataset.from_tensors({'image': tf.constant(np.random.rand(*shape), dtype=tf.float32)})
        ds = nl.dataset.labelled({'0': tsr_ds}, label_key="label").batch(1)

        experiment = nl.train.NLExperiment(
            self._experiment_path,
            model,
            eval_dataset=ds,
            train_dataset=ds,
            hyperparameters={'learning_rate': 0},
            seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)

    def test_serialization_input_rangenorm(self):
        model = TestInputBModel()

        shape = model.input_shape
        tsr_ds = tf.data.Dataset.from_tensors({'image': tf.constant(np.random.rand(*shape), dtype=tf.float32)})
        ds = nl.dataset.labelled({'0': tsr_ds}, label_key="label").batch(1)

        experiment = nl.train.NLExperiment(
            self._experiment_path,
            model,
            eval_dataset=ds,
            train_dataset=ds,
            hyperparameters={'learning_rate': 0},
            seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)

    def test_serialization_dense(self):
        model = TestDenseModel()

        ds = nl.dataset.labelled({'0': nl.dataset.ones(model.input_shape)}, label_key="label").batch(1)

        experiment = nl.train.NLExperiment(
            self._experiment_path,
            model,
            eval_dataset=ds,
            train_dataset=ds,
            hyperparameters={'learning_rate': 0},
            seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)

    def test_serialization_avg_pooling(self):
        model = TestAvgPoolingModel()

        ds = nl.dataset.labelled({'0': nl.dataset.ones(model.input_shape)}, label_key="label").batch(1)

        experiment = nl.train.NLExperiment(
            self._experiment_path,
            model,
            eval_dataset=ds,
            train_dataset=ds,
            hyperparameters={'learning_rate': 0},
            seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)

    def test_serialization_max_pooling(self):
        model = TestMaxPoolingModel()

        ds = nl.dataset.labelled({'0': nl.dataset.ones(model.input_shape)}, label_key="label").batch(1)

        experiment = nl.train.NLExperiment(
            self._experiment_path,
            model,
            eval_dataset=ds,
            train_dataset=ds,
            hyperparameters={'learning_rate': 0},
            seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)

    def test_serialization_conv2d_5(self):
        model = TestConv2DModel(5)

        ds = nl.dataset.labelled({'0': nl.dataset.ones(model.input_shape)}, label_key="label").batch(1)

        experiment = nl.train.NLExperiment(
            self._experiment_path,
            model,
            eval_dataset=ds,
            train_dataset=ds,
            hyperparameters={'learning_rate': 0},
            seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)

    def test_serialization_conv2d_1(self):
        model = TestConv2DModel(1)

        ds = nl.dataset.labelled({'0': nl.dataset.ones(model.input_shape)}, label_key="label").batch(1)

        experiment = nl.train.NLExperiment(
            self._experiment_path,
            model,
            eval_dataset=ds,
            train_dataset=ds,
            hyperparameters={'learning_rate': 0},
            seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)

    def test_serialization_conv2d_2(self):
        model = TestConv2DModel(2)

        ds = nl.dataset.labelled({'0': nl.dataset.ones(model.input_shape)}, label_key="label").batch(1)

        experiment = nl.train.NLExperiment(
            self._experiment_path,
            model,
            eval_dataset=ds,
            train_dataset=ds,
            hyperparameters={'learning_rate': 0},
            seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)

    def test_serialization_conv2d_3(self):
        model = TestConv2DModel(3)

        ds = nl.dataset.labelled({'0': nl.dataset.ones(model.input_shape)}, label_key="label").batch(1)

        experiment = nl.train.NLExperiment(
            self._experiment_path,
            model,
            eval_dataset=ds,
            train_dataset=ds,
            hyperparameters={'learning_rate': 0},
            seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)

    def test_serialization_separable_conv2d(self):
        model = TestSeparableConv2DModel(3)

        ds = nl.dataset.labelled({'0': nl.dataset.ones(model.input_shape)}, label_key="label").batch(1)

        experiment = nl.train.NLExperiment(
            self._experiment_path,
            model,
            eval_dataset=ds,
            train_dataset=ds,
            hyperparameters={'learning_rate': 0},
            seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)


    def test_serialization_dropout(self):
        model = TestDropoutModel()

        ds = nl.dataset.labelled({'0': nl.dataset.ones(model.input_shape)}, label_key="label").batch(1)

        experiment = nl.train.NLExperiment(
            self._experiment_path,
            model,
            eval_dataset=ds,
            train_dataset=ds,
            hyperparameters={'learning_rate': 0},
            seed=42)

        experiment.train(steps=1)
        self.assert_tf_export(experiment)


if __name__ == '__main__':
    unittest.main()
