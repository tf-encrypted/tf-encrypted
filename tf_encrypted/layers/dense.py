# pylint: disable=arguments-differ
"""Dense (i.e. fully connected) Layer implementation."""
from typing import Optional
from typing import Union

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.layers.core import Layer
from tf_encrypted.protocol import TFETensor

InitialTensor = Optional[Union[np.ndarray, tf.Tensor, TFETensor]]


class Dense(Layer):
    """Standard dense linear layer including bias.

    :param int in_features: number of input features
    :param int out_features: number of output neurons for the layer
    """

    def __init__(
        self, input_shape, out_features, transpose_input=False, transpose_weight=False,
    ):
        self.in_features = input_shape[-1]
        self.out_features = out_features

        self.layer_input = None
        self.weights = None
        self.bias = None

        self.transpose_input = transpose_input
        self.transpose_weight = transpose_weight

        super(Dense, self).__init__(input_shape)

    def get_output_shape(self):
        return [self.input_shape[0] + self.out_features]

    def initialize(
        self, initial_weights: InitialTensor = None, initial_bias: InitialTensor = None,
    ) -> None:
        if initial_weights is None:
            initial_size = (self.in_features, self.out_features)
            initial_weights = np.random.normal(scale=0.1, size=initial_size)
        if initial_bias is not None:
            self.bias = tfe.define_private_variable(initial_bias)

        self.weights = tfe.define_private_variable(initial_weights)

        if self.transpose_weight:
            self.weights = self.weights.transpose()

    def forward(self, x):
        self.layer_input = x

        if self.transpose_input:
            self.layer_input = self.layer_input.transpose()

        if self.bias:
            y = x.matmul(self.weights) + self.bias
        else:
            y = x.matmul(self.weights)
        return y

    def backward(self, d_y, learning_rate):
        x = self.layer_input
        if self.transpose_input:
            self.layer_input = self.layer_input.transpose()

        d_x = d_y.matmul(self.weights.transpose())

        d_weights = x.transpose().matmul(d_y)
        self.weights.assign((d_weights * learning_rate).neg() + self.weights)

        if self.bias:
            d_bias = d_y.reduce_sum(axis=0)
            self.bias -= d_bias * learning_rate

        return d_x
