import numpy as np
import math
from . import core
from ..protocol.pond import PondPublicVariable, PondPrivateVariable
from ..protocol.types import TFEVariable
import tensorflow as tf
from typing import Optional, Union, Tuple, NewType, List

IntTuple = Union[int, Tuple[int, int], List[int]]


class AveragePooling2D(core.Layer):
    """AveragePooling2D layer
    Arguments:
    pool_size (int or tuple/list of ints) size of kernel
    strides (int or tuple/list of ints) stride length
    padding (str) SAME or VALID
    """
    def __init__(self,
                 input_shape: List[int],
                 pool_size: IntTuple,
                 strides: Optional[IntTuple] = None,
                 padding: str = "SAME") -> None:
        if type(pool_size) == int:
            pool_size = (pool_size, pool_size)  # type: ignore
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        elif type(strides) == int:
            strides = (strides, strides)  # type: ignore
        self.strides = strides
        if padding not in ['SAME', 'VALID']:
            raise ValueError("Don't know how to do padding of type {}".format(padding))
        if padding == 'VALID':
            raise NotImplementedError('Padding not supported')
        self.padding = padding
        super(AveragePooling2D, self).__init__(input_shape)
        self.cache = None
        self.cached_input_shape = None

    def initialize(self,
                   input_shape: IntTuple,
                   initializer: Optional[TFEVariable] = None) -> None:
        pass

    def get_output_shape(self) -> List[int]:
        channels = self.input_shape[1]  # assumes NCHW
        H_in = self.input_shape[2]
        W_in = self.input_shape[3]
        if self.padding == "SAME":
            H_out: int = math.ceil(H_in / self.strides[0])
            W_out: int = math.ceil(W_in / self.strides[1])
        else:
            H_out = math.ceil((H_in - self.pool_size[0] + 1) / self.strides[0])
            W_out = math.ceil((W_in - self.pool_size[1] + 1) / self.strides[1])
        return [self.input_shape[0], self.input_shape[1], H_out, W_out]

    def forward(self, x: TFEVariable) -> TFEVariable:
        self.cached_input_shape = x.shape
        self.cache = x
        return self.prot.avgpool2d(x, self.pool_size, self.strides, self.padding)

    def backward(self, d_y: TFEVariable, learning_rate: float) -> Optional[TFEVariable]:
        raise NotImplementedError
