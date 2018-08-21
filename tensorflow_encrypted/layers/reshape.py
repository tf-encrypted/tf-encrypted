from . import core

from typing import List


class Reshape(core.Layer):
    def __init__(self, input_shape: List[int], output_shape: List[int] = [-1]) -> None:
        self.output_shape = output_shape

        super(Reshape, self).__init__(input_shape)

    def get_output_shape(self) -> List[int]:
        return self.output_shape

    def initialize(self, *args, **kwargs) -> None:
        pass

    def forward(self, x):
        y = self.prot.reshape(x, self.output_shape)
        self.layer_output = y
        return y

    def backward(self, *args, **kwargs):
        raise NotImplementedError
