import numpy as np
from . import core


class Dense(core.Layer):
    """Standard dense linear layer including bias.

    Arguments:
    in_features (int, required): number of input features
    out_features (int, required): number of output neurons for the layer
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features

        self.layer_input = None
        self.weights = None
        self.bias = None

    def initialize(self, initial_weights=None, initial_bias=None):
        if initial_weights is None:
            initial_size = (self.in_features, self.out_features)
            initial_weights = np.random.normal(scale=0.1, size=initial_size)
        if initial_bias is None:
            initial_bias = np.zeros((1, self.out_features))

        self.weights = self.prot.define_private_variable(initial_weights)
        self.bias = self.prot.define_private_variable(initial_bias)

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
