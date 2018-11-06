import numpy as np
import tensorflow as tf

from typing import List, Union, Optional

from . import core
from ..protocol.pond import PondPublicTensor, PondPrivateTensor


class Dense(core.Layer):
    """Standard dense linear layer including bias.

    :param int in_features: number of input features
    :param int out_features: number of output neurons for the layer
    """

    def __init__(self, input_shape: List[int], out_features: int) -> None:
        self.in_features = input_shape[-1]
        self.out_features = out_features

        self.layer_input = None
        self.weights = None
        self.bias = None

        super(Dense, self).__init__(input_shape)

    def get_output_shape(self):
        return [self.input_shape[0] + self.out_features]

    def initialize(
        self,
        initial_weights: Optional[Union[np.ndarray, tf.Tensor, PondPublicTensor, PondPrivateTensor]]=None,
        initial_bias: Optional[Union[np.ndarray, tf.Tensor, PondPublicTensor, PondPrivateTensor]]=None
    ) -> None:
        if initial_weights is None:
            initial_size = (self.in_features, self.out_features)
            initial_weights = np.random.normal(scale=0.1, size=initial_size)
        if initial_bias is None:
            initial_bias = np.zeros((1, self.out_features))

        self.weights = self.prot.define_private_variable(initial_weights)
        self.bias = self.prot.define_private_variable(initial_bias)

    def forward(self, x):
        self.layer_input = x
        y = x.matmul(self.weights) + self.bias
        return y

    def backward(self, d_y, learning_rate):
        x = self.layer_input
        d_x = d_y.matmul(self.weights.transpose())

        d_weights = x.transpose().matmul(d_y)
        d_bias = d_y.reduce_sum(axis=0)

        self.weights.assign((d_weights * learning_rate).neg() + self.weights)
        self.bias.assign((d_bias * learning_rate).neg() + self.bias)

        return d_x
