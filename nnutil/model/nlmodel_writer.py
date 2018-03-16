import numpy as np
import tensorflow as tf


class NLModelWriter():
    def __init__(self, model, sess):
        self._model = model
        self._sess = sess

        self._name = model.name
        self._shape = model.shape
        self._layers = model.layers
        self._nlayers = len(model.layers)
        self._labels = model.labels
        self._nlabels = len(model.labels)


    def write_scalar(self, x, fd):
        """Write a single scalar in the .wgt file"""
        fd.write('%f' % x)
        fd.write("\n")


    def write_vector(self, vec, fd):
        """ Write a column vector in the .wgt file"""
        shape = vec.shape
        for i in range(0, vec.shape[0]):
            fd.write('%f' % vec[i])
            fd.write("\n")


    def write_matrix(self, mat, fd):
        """Write a matrix in the .wgt file"""
        shape = mat.shape
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                fd.write('%f' % mat[i,j])
                fd.write(" ")
            fd.write("\n")


    def activation_function(self, layer):
        if layer.activation == tf.nn.relu:
            return "relu"

        elif layer.activation == tf.identity or layer.activation is None:
            return "identity"

        else:
            raise Exception("Unhandled activation function: {}".format(str(layer.activation)))


    def write_net_preamble(self, nlayers, fd):
        if fd is None:
            return

        fd.write("<NET>\n")
        fd.write("NetID [{}]\n".format(self._name))
        fd.write("NumLayers [{}]\n".format(nlayers))
        fd.write("InputSize [{} {}]\n".format(self._shape[0], self._shape[1]))
        fd.write("OutputSize [{}]\n".format(self._nlabels))
        fd.write("</NET>\n")
        fd.write("\n")


    def write_net_input(self, nlayer, name, input_shape, fd):
        if fd is None:
            return

        fd.write("<LAYER>\n")
        fd.write("Layer{} [{}]\n".format(nlayer, name))
        fd.write("Type [I]\n")
        fd.write("OutputSize [{} {}]\n".format(input_shape[0], input_shape[1]))
        fd.write("NumFMaps [{}]\n".format(input_shape[2]))
        fd.write("NormMethod [none]\n")
        fd.write("</LAYER>\n")
        fd.write("\n")


    def write_net_conv2d(self, nlayer, layer, input_shape, output_shape, kernel_shape, fd):
        if fd is None:
            return

        fd.write("<LAYER>\n")
        fd.write("Layer{} [{}]\n".format(nlayer, layer.name))
        fd.write("Type [C]\n")
        fd.write("InputSize [{} {}]\n".format(input_shape[0], input_shape[1]))
        fd.write("OutputSize [{} {}]\n".format(output_shape[0], output_shape[1]))
        fd.write("KernelSize [{} {}]\n".format(kernel_shape[0], kernel_shape[1]))
        fd.write("ActFunction [{}]\n".format(self.activation_function(layer)))
        fd.write("NumFMaps [{}]\n".format(kernel_shape[3]))
        fd.write("Connections\n")
        for i in range(0, output_shape[2]):
            fd.write("[{} {}]\n".format(input_shape[2], ' '.join([str(i) for i in range(0, input_shape[2])])))
        fd.write("</LAYER>\n")
        fd.write("\n")


    def write_wgt_conv2d(self, kernel, bias, fd):
        if fd is None:
            return

        for fout in range(0, kernel.shape[3]):
            for fin in range(0, kernel.shape[2]):
                self.write_matrix(kernel[:,:,fin,fout], fd)
                fd.write("\n")
            self.write_scalar(bias[fout], fd)
            fd.write("\n")


    def write_net_max_pooling(self, nlayer, layer, input_shape, output_shape, pool_size, fd):
        if fd is None:
            return

        fd.write("<LAYER>\n")
        fd.write("Layer{} [{}]\n".format(nlayer, layer.name))
        fd.write("Type [MP]\n")
        fd.write("InputSize [{} {}]\n".format(input_shape[0], input_shape[1]))
        fd.write("OutputSize [{} {}]\n".format(output_shape[0], output_shape[1]))
        fd.write("KernelSize [{} {}]\n".format(pool_size[0], pool_size[1]))
        fd.write("ActFunction [identity]\n")
        fd.write("NumFMaps [{}]\n".format(input_shape[2]))
        fd.write("Connections\n")
        for i in range(0, input_shape[2]):
            fd.write("[1 {}]\n".format(i))
        fd.write("</LAYER>\n")
        fd.write("\n")


    def write_wgt_max_pooling(self, output_shape, pool_size, fd):
        if fd is None:
            return

        for i in range(0, output_shape[-1]):
            fd.write("1.0\n\n0.0\n\n")


    def write_net_avg_pooling(self, nlayer, layer, input_shape, output_shape, pool_size, fd):
        if fd is None:
            return

        fd.write("<LAYER>\n")
        fd.write("Layer{} [{}]\n".format(nlayer, layer.name))
        fd.write("Type [S]\n")
        fd.write("InputSize [{} {}]\n".format(input_shape[0], input_shape[1]))
        fd.write("OutputSize [{} {}]\n".format(output_shape[0], output_shape[1]))
        fd.write("KernelSize [{} {}]\n".format(pool_size[0], pool_size[1]))
        fd.write("ActFunction [identity]\n")
        fd.write("NumFMaps [{}]\n".format(input_shape[2]))
        fd.write("Connections\n")
        for i in range(0, input_shape[2]):
            fd.write("[1 {}]\n".format(i))
        fd.write("</LAYER>\n")
        fd.write("\n")


    def write_wgt_avg_pooling(self, output_shape, pool_size, fd):
        if fd is None:
            return

        for i in range(0, output_shape[-1]):
            fd.write("{}\n\n0.0\n\n".format(1.0/np.prod(pool_size)))


    def write_net_dense(self, nlayer, layer, input_shape, output_shape, kernel_shape, fd):
        if fd is None:
            return

        fd.write("<LAYER>\n")
        fd.write("Layer{} [{}]\n".format(nlayer, layer.name))
        fd.write("Type [F]\n")
        fd.write("NumInputs [{}]\n".format(kernel_shape[0]))
        fd.write("NumOutputs [{}]\n".format(kernel_shape[1]))
        fd.write("ActFunction [{}]\n".format(self.activation_function(layer)))
        fd.write("NumFMaps [1]\n")
        fd.write("Connections\n")
        fd.write("[{} {}]\n".format(int(input_shape[-1]), ' '.join([str(i) for i in range(0, int(input_shape[-1]))])))
        fd.write("</LAYER>\n")
        fd.write("\n")


    def write_wgt_dense(self, kernel, bias, fd):
        if fd is None:
            return

        self.write_matrix(np.transpose(kernel), fd)
        fd.write("\n")
        self.write_vector(bias, fd)
        fd.write("\n")


    def write_net_output(self, nlayer, layer, input_shape, output_shape, kernel_shape, fd):
        if fd is None:
            return

        # TODO: asserts on input and output shapes
        assert(kernel_shape[1] == self._nlabels)

        fd.write("<LAYER>\n")
        fd.write("Layer{} [{}]\n".format(nlayer, layer.name))
        fd.write("Type [O]\n")
        fd.write("NumInputs [{}]\n".format(kernel_shape[0]))
        fd.write("NumOutputs [{}]\n".format(kernel_shape[1]))
        fd.write("ActFunction [{}]\n".format(self.activation_function(layer)))
        fd.write("OutFunction [softmax]\n")
        fd.write("LossFunction [cross_entropy]\n")
        fd.write("NumFMaps [1]\n")
        fd.write("Labels [{}]\n".format(';'.join(self._labels)))
        fd.write("Connections\n")
        fd.write("[{} {}]\n".format(int(input_shape[-1]), ' '.join([str(i) for i in range(0, int(input_shape[-1]))])))
        fd.write("</LAYER>\n")
        fd.write("\n")


    def write_wgt_output(self, kernel, bias, fd):
        if fd is None:
            return

        self.write_wgt_dense(kernel, bias, fd)


    def write(self, net_fd, wgt_fd):

        # TODO: do not hard-code skipped layers here
        nlayers = len([l for l in self._layers if type(l) not in set([tf.layers.Flatten, tf.layers.Dropout])]) + 1

        shape = tuple(self._shape)

        self.write_net_preamble(nlayers, net_fd)
        self.write_net_input(0, "input", shape, net_fd)

        nlayer = 1

        # TODO: extract input shape from the layer instead of passing it around and doing shape inference
        # on tensorflow's back. Somehow, at this point we can't do l.input_shape value is not available.
        for i, layer in enumerate(self._layers):
            input_shape = tuple(shape)

            if i == len(self._layers) - 1:
                if type(layer) != tf.layers.Dense:
                    raise Exception("Last layer must be a tf.layers.Dense layer")

                kernel, bias = self._sess.run([layer.kernel, layer.bias])

                kernel_shape = kernel.shape
                output_shape = (int(kernel_shape[1]),)

                self.write_net_output(nlayer, layer, input_shape, output_shape, kernel_shape, net_fd)
                self.write_wgt_output(kernel, bias, wgt_fd)

                nlayer += 1

            elif type(layer) == tf.layers.Conv2D:
                kernel, bias = self._sess.run([layer.kernel, layer.bias])

                kernel_shape = kernel.shape
                output_shape = (shape[0] - int(kernel_shape[0]) + 1, shape[1] - int(kernel_shape[1]) + 1, int(kernel_shape[3]))

                self.write_net_conv2d(nlayer, layer, input_shape, output_shape, kernel_shape, net_fd)
                self.write_wgt_conv2d(kernel, bias, wgt_fd)

                nlayer += 1

            elif type(layer) == tf.layers.Dense:
                kernel, bias = self._sess.run([layer.kernel, layer.bias])

                kernel_shape = kernel.shape
                output_shape = (int(kernel_shape[1]),)

                self.write_net_dense(nlayer, layer, input_shape, output_shape, kernel_shape, net_fd)
                self.write_wgt_dense(kernel, bias, wgt_fd)

                nlayer += 1

            elif type(layer) == tf.layers.MaxPooling2D:
                pool_size = layer.pool_size
                output_shape = (int(input_shape[0] / pool_size[0]), int(input_shape[1] / pool_size[1]), input_shape[2])

                self.write_net_max_pooling(nlayer, layer, input_shape, output_shape, pool_size, net_fd)
                self.write_wgt_max_pooling(output_shape, pool_size, wgt_fd)

                nlayer += 1

            elif type(layer) == tf.layers.AveragePooling2D:
                pool_size = layer.pool_size
                output_shape = (int(input_shape[0] / pool_size[0]), int(input_shape[1] / pool_size[1]), input_shape[2])

                self.write_net_avg_pooling(nlayer, layer, input_shape, output_shape, pool_size, net_fd)
                self.write_wgt_avg_pooling(output_shape, pool_size, wgt_fd)

                nlayer += 1

            elif type(layer) == tf.layers.Flatten:
                output_shape = (input_shape[2],)

                assert(len(input_shape) == 3)
                assert(input_shape[0:2] == (1, 1))

            elif type(layer) == tf.layers.Dropout:
                output_shape = input_shape

            else:
                raise Exception("Unhandled layer type: {}".format(str(type(layer))))

            shape = output_shape
