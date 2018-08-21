from . import core

from typing import List, Any


class Reshape(core.Layer):
    def __init__(self, shape: List[int] = [-1]) -> None:
        self.shape: List[int] = shape
        self.layer_output: Any = None

    def initialize(self, *args, **kwargs) -> None:
        pass

    def forward(self, x):
        y = self.prot.reshape(x, self.shape)
        self.layer_output = y
        return y

    def backward(self, *args, **kwargs):
        raise NotImplementedError
