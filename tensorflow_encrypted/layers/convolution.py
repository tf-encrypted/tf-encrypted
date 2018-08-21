import numpy as np
from . import core


class Conv2D(core.Layer):
    def __init__(self, filter_shape, strides=1, padding="SAME",
                 filter_init=lambda shp: np.random.normal(scale = 0.1, size = shp),
                 l2reg_lambda=0.0, channels_first=True):
        """ 2 Dimensional convolutional layer, expects NCHW data format
            filter_shape: tuple of rank 4
            strides: int with stride size
            filter init: lambda function with shape parameter
            Example:
            Conv2D((4, 4, 1, 20), strides=2, filter_init=lambda shp:
                    np.random.normal(scale=0.01, size=shp))
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

    def initialize(self, input_shape, initial_weights=None):

        h_filter, w_filter, d_filters, n_filters = self.fshape
        n_x, d_x, h_x, w_x = input_shape

        if self.padding == "SAME":
            h_out = int(np.ceil(float(h_x) / float(self.strides)))
            w_out = int(np.ceil(float(w_x) / float(self.strides)))
        if self.padding == "VALID":
            h_out = int(np.ceil(float(h_x - h_filter + 1) / float(self.strides)))
            w_out = int(np.ceil(float(w_x - w_filter + 1) / float(self.strides)))

        if initial_weights is None:
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
        h_filter, w_filter, d_filter, n_filter = map(int, self.weights.shape)

        if self.model.layers.index(self) != 0:
            W_reshaped = self.weights.reshape(n_filter, -1).transpose()
            dout_reshaped = d_y.transpose(1, 2, 3, 0).reshape(n_filter, -1)
            dx = W_reshaped.dot(dout_reshaped).col2im(imshape=self.cached_input_shape,
                                                      field_height=h_filter,
                                                      field_width=w_filter,
                                                      padding=self.padding,
                                                      stride=self.strides)

        d_w = self.prot.conv2d_bw(x, d_y, self.weights.shape, self.strides, self.padding)
        d_bias = d_y.sum(axis=0)

        self.weights.assign((d_w * learning_rate).neg() + self.weights)
        self.bias.assign((d_bias * learning_rate).neg() + self.bias)

        return dx


def set_protocol(new_prot):
    core.Layer.prot = new_prot
