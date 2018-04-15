import os
import tensorflow as tf

from tensorflow.python.framework.tensor_spec import TensorSpec

class Mobilenet(tf.layers.Layer):
    def __init__(self, layer=None, **kwargs):
        super(Mobilenet, self).__init__(**kwargs)
        self._path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'mobilenet_v1_1.0_224_frozen.pb')

        tpl = "MobilenetV1/MobilenetV1/{}_pointwise/Relu6"
        self._layer_spec = [
            TensorSpec((224, 224, 3), tf.float32, name='input'),
            TensorSpec((112, 112, 64), tf.float32, name=tpl.format("Conv2d_1")),
            TensorSpec((56, 56, 128), tf.float32, name=tpl.format("Conv2d_2")),
            TensorSpec((56, 56, 128), tf.float32, name=tpl.format("Conv2d_3")),
            TensorSpec((28, 28, 256), tf.float32, name=tpl.format("Conv2d_4")),
            TensorSpec((28, 28, 256), tf.float32, name=tpl.format("Conv2d_5")),
            TensorSpec((14, 14, 512), tf.float32, name=tpl.format("Conv2d_6")),
            TensorSpec((14, 14, 512), tf.float32, name=tpl.format("Conv2d_7")),
            TensorSpec((14, 14, 512), tf.float32, name=tpl.format("Conv2d_8")),
            TensorSpec((14, 14, 512), tf.float32, name=tpl.format("Conv2d_9")),
            TensorSpec((14, 14, 512), tf.float32, name=tpl.format("Conv2d_10")),
            TensorSpec((14, 14, 512), tf.float32, name=tpl.format("Conv2d_11")),
            TensorSpec((7, 7, 1024), tf.float32, name=tpl.format("Conv2d_12")),
            TensorSpec((7, 7, 1024), tf.float32, name=tpl.format("Conv2d_13"))
        ]

        if layer is None:
            layer = -1

        if layer < 0:
            layer = len(self._layer_spec) + layer

        self._output_layer = layer

        self._graph_def = None
        self._input_shape = (224, 224, 3)
        self._output_shape = tuple(self._layer_spec[self._output_layer].shape)

    def build(self, input_shape):
        with open(self._path, 'rb') as fd:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fd.read())

        self._graph_def = graph_def

    def call(self, inputs):
        if tuple(inputs.shape[1:]) != self._input_shape:
            raise Exception("Input shape does not match")

        results = tf.import_graph_def(
            self._graph_def,
            input_map={'input': inputs},
            return_elements=[ls.name for ls in self._layer_spec],
            name=self.name
        )

        self._outputs = [op.outputs[0] for op in results]

        for i, (x, ls) in enumerate(zip(self._outputs, self._layer_spec)):
            if x.shape[1:] != ls.shape:
                raise Exception("Output shape does not match layer:\n layer: {}\n name: {}\n spec: {}\n tensor: {}".format(
                    i, ls.name, ls.shape.as_list(), x.shape.as_list()[1:]))

        return self._outputs[self._output_layer]

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + tuple(self._layer_spec[self._output_layer].shape)

    @property
    def layer_activations(self):
        return self._outputs
