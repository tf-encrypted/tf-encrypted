import numpy as np
import math
from . import core
from ..protocol.pond import PondPublicVariable, PondPrivateVariable
from ..protocol.types import TFEVariable
import tensorflow as tf
from typing import Optional, Union, Tuple, NewType

IntTuple = Union[int, Tuple[int]]


class AveragePooling2D(core.Layer):
    """AveragePooling2D layer
    Arguments:
    pool_size (int or tuple of ints) size of kernel
    strides (int or tuple of ints)
    padding (str)
    """
    def __init__(self,
                 pool_size: IntTuple,
                 strides: Optional[IntTuple] = None,
                 padding: str = "SAME") -> None:
        super(AveragePooling2D, self).__init__()
        if type(pool_size) == int:
            pool_size = (pool_size, pool_size)  # type: ignore
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        elif type(strides) == int:
            strides = (strides, strides)  # type: ignore
        self.strides = strides
        self.padding = padding
        self.cache = None
        self.cached_input_shape = None

    def initialize(self,
                   input_shape: Tuple[int],
                   initializer: Optional[TFEVariable] = None) -> None:
        pass

    def forward(self, x: TFEVariable) -> TFEVariable:
        self.cached_input_shape = x.shape
        self.cache = x
        return self.prot.avgpool2d(x, self.pool_size, self.strides, self.padding)

    def backward(self, d_y: TFEVariable, learning_rate: float) -> Optional[TFEVariable]:
        raise NotImplementedError
