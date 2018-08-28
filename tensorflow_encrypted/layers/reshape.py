from . import core

from typing import List, Optional

from ..protocol.pond import PondPrivateTensor


class Reshape(core.Layer):
    def __init__(self, shape: List[int] = [-1]) -> None:
        self.shape = shape
        self.layer_output: Optional[PondPrivateTensor] = None

    def initialize(self, *args, **kwargs) -> None:
        pass

    def forward(self, x: PondPrivateTensor) -> PondPrivateTensor:
        y = self.prot.reshape(x, self.shape)
        self.layer_output = y
        return y

    def backward(self, *args, **kwargs):
        raise NotImplementedError
