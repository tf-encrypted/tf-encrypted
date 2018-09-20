import numpy as np
from . import core

from typing import List
from tensorflow_encrypted.protocol.pond import PondPrivateTensor
from tensorflow_encrypted.layers.activation import Tanh, Sigmoid


# Ressource: http://chris-chris.ai/2017/10/10/LSTM-LayerNorm-breakdown-eng/
class LSTM(core.Layer):
    def __init__(self, input_shape, prev_cell, f, i, o, j) -> None:
        self.input_shape = None
        self.prev_cell = prev_cell
        self.f = f
        self.i = i
        self.o = o
        self.j = j

        super(LSTM, self).__init__(input_shape)

    def get_output_shape(self) -> List[int]:
        return self.input_shape

    def initialize(self) -> None:
        pass


    def forward(self) -> PondPrivateTensor:

        forget_bias = 1.0
        input_shape = [2]

        tanh_layer = Tanh(input_shape)
        sigmoid_layer = Sigmoid(input_shape)

        sigmoid_f = sigmoid_layer.forward(self.f + forget_bias)
        sigmoid_i = sigmoid_layer.forward(self.i)
        sigmoid_o = sigmoid_layer.forward(self.o)
        tanh_g = tanh_layer.forward(self.j)

        new_c = self.prev_cell * sigmoid_f + sigmoid_i  * tanh_g

        new_h = tanh_layer.forward(new_c) * sigmoid_o

        # How to eval tuples with tf-encrypted
        return new_h


    def backward(self) -> None:
        raise NotImplementedError
