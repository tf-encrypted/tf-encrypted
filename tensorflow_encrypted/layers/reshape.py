import numpy as np
from typing import List

from . import core


class Reshape(core.Layer):
    def __init__(self, input_shape: List[int], output_shape: List[int] = [-1]) -> None:
        self.output_shape = output_shape

        super(Reshape, self).__init__(input_shape)

    def get_output_shape(self) -> List[int]:
        if -1 in self.output_shape:
            total_input_dims = np.prod(self.input_shape)

            dim = 1
            for i in self.output_shape:
                if i != -1:
                    dim *= i
            missing_dim = int(total_input_dims / dim)

            output_shape = self.output_shape
            for key, i in enumerate(output_shape):
                if i == -1:
                    output_shape[key] = missing_dim

            return output_shape
        else:
            return self.output_shape

    def initialize(self, *args, **kwargs) -> None:
        pass

    def forward(self, x):
        y = self.prot.reshape(x, self.output_shape)
        self.layer_output = y
        return y

    def backward(self, *args, **kwargs):
        raise NotImplementedError
