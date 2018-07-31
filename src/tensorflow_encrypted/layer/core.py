import numpy as np
import math
from ..protocol.pond import Pond

DEFAULT_PROTOCOL = Pond

# TODO
# split backward function in compute_gradient and compute_backpropagated_error ?

class Layer:
    prot = DEFAULT_PROTOCOL


class Dense(Layer):

    def __init__(self, num_nodes, num_inputs):

        self.num_nodes = num_nodes
        self.num_inputs = num_inputs

        self.layer_input = None
        self.weights = None
        self.bias = None

    def initialize(self):
        initial_weights = np.random.normal(scale=0.1, size=(self.num_inputs, self.num_nodes))
        self.weights = self.prot.define_private_variable(initial_weights)
        self.bias = self.prot.define_private_variable(np.zeros(1, self.num_nodes))

    def forward(self, x):
        self.layer_input = x
        y = x.dot(self.weights) + self.bias
        return y

    def backward(self, d_y, learning_rate):
        x = self.layer_input
        d_x = d_y.dot(self.weights.transpose())

        d_weights = x.transpose().dot(d_y)
        d_bias = d_y.sum(axis=0)

        self.weights.assign((d_weights * learning_rate).neg() + self.weights)
        self.bias.assign((d_bias * learning_rate).neg() + self.bias)

        return d_x


class Sigmoid(Layer):

    def __init__(self):
        self.layer_output = None

    def initialize(self, input_shape, initializer=None):
        pass

    def forward(self, x):
        y = self.prot.sigmoid(x)
        self.layer_output = y
        return y

    def backward(self, d_y, *args):
        y = self.layer_output
        d_x = d_y * y * (y.neg() + 1)
        return d_x



class Conv2D(Layer):
    def __init__(self, filter_shape, strides=1, padding="SAME",
                 filter_init=lambda shp: np.random.normal(scale=0.1, size=shp),
                 l2reg_lambda=0.0, channels_first=True):
        """ 2 Dimensional convolutional layer, expects NCHW data format
            filter_shape: tuple of rank 4
            strides: int with stride size
            filter init: lambda function with shape parameter
            Example: Conv2D((4, 4, 1, 20), strides=2, filter_init=lambda shp: np.random.normal(scale=0.01,
            size=shp))
        """
        self.fshape = filter_shape
        self.strides = strides
        self.padding = padding
        self.filter_init = filter_init
        self.l2reg_lambda = l2reg_lambda
        self.cache = None
        self.cached_x_col = None
        self.cached_input_shape = None
        self.initializer = None
        self.weights = None
        self.bias = None
        self.model = None
        assert channels_first

    def initialize(self, input_shape):

        h_filter, w_filter, d_filters, n_filters = self.fshape
        n_x, d_x, h_x, w_x = input_shape

        if self.padding == "SAME":
            h_out = int(math.ceil(float(h_x) / float(self.strides)))
            w_out = int(math.ceil(float(w_x) / float(self.strides)))
        if self.padding == "VALID":
            h_out = int(math.ceil(float(h_x - h_filter + 1) / float(self.strides)))
            w_out = int(math.ceil(float(w_x - w_filter + 1) / float(self.strides)))

        initial_weights = self.filter_init(self.fshape)
        self.weights = self.prot.define_private_variable(initial_weights)
        self.bias = self.prot.define_private_variable(np.zeros((n_filters, h_out, w_out)))

        return [n_x, n_filters, h_out, w_out]

    def forward(self, x):
        self.cached_input_shape = x.shape
        self.cache = x
        out = self.prot.conv2d(x, self.weights, self.strides, self.padding)

        return out + self.bias

    def backward(self, d_y, learning_rate):
        x = self.cache
        h_filter, w_filter, d_filter, n_filter = self.filters.shape
        dx = None

        if self.model.layers.index(self) != 0:
            W_reshaped = self.filters.reshape(n_filter, -1).transpose()
            dout_reshaped = d_y.transpose(1, 2, 3, 0).reshape(n_filter, -1)
            dx = W_reshaped.dot(dout_reshaped).col2im(imshape=self.cached_input_shape, field_height=h_filter,
                                                      field_width=w_filter, padding=self.padding, stride=self.strides)

        d_w = self.prot.conv2d_bw(x, d_y, self.filters.shape, self.strides, self.padding)
        d_bias = d_y.sum(axis=0)

        self.filters.assign((d_w * learning_rate).neg() + self.filters)
        self.bias.assign((d_bias * learning_rate).neg() + self.bias)

        return dx


def set_protocol(new_prot):
    Layer.prot = new_prot
