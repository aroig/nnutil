import os
import tensorflow as tf

class Mobilenet(tf.layers.Layer):
    def __init__(self, **kwargs):
        super(Mobilenet, self).__init__(**kwargs)
        self._path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'mobilenet_v1_1.0_224_frozen.pb')

        self._graph_def = None
        self._input_shape = (224, 224, 3)
        self._output_shape = (1001,)

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
            return_elements=["MobilenetV1/Predictions/Softmax"],
            name=self.name
        )

        output = results[0].outputs[0]

        if tuple(output.shape[1:]) != self._output_shape:
            raise Exception("Output shape does not match")

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self._output_shape
