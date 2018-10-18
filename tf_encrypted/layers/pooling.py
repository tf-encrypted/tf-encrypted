import math
from . import core

from ..protocol.pond import TFEVariable
from typing import Optional, Union, Tuple, List
from abc import abstractmethod

IntTuple = Union[int, Tuple[int, int], List[int]]


class Pooling2D(core.Layer):
    """
    Subclass for Average and Max pooling layers

    Do not instantiate.
    """
    def __init__(self,
                 input_shape: List[int],
                 pool_size: IntTuple,
                 strides: Optional[IntTuple] = None,
                 padding: str = "SAME", channels_first: bool = True) -> None:
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
        self.padding = padding
        self.channels_first = channels_first

        super(Pooling2D, self).__init__(input_shape)
        self.cache = None
        self.cached_input_shape = None

    def initialize(self,
                   input_shape: IntTuple,
                   initializer: Optional[TFEVariable] = None) -> None:
        pass

    def get_output_shape(self) -> List[int]:
        if self.channels_first:
            _, channels, H_in, W_in = self.input_shape
        else:
            _, H_in, W_in, channels = self.input_shape

        if self.padding == "SAME":
            H_out = math.ceil(H_in / self.strides[0])
            W_out = math.ceil(W_in / self.strides[1])
        else:
            H_out = math.ceil((H_in - self.pool_size[0] + 1) / self.strides[0])
            W_out = math.ceil((W_in - self.pool_size[1] + 1) / self.strides[1])
        return [self.input_shape[0], self.input_shape[1], H_out, W_out]

    @abstractmethod
    def pool(self, x: TFEVariable, pool_size, strides, padding) -> TFEVariable:
        raise NotImplementedError

    def forward(self, x: TFEVariable) -> TFEVariable:
        if not self.channels_first:
            x = self.prot.transpose(x, perm=[0, 3, 1, 2])

        self.cached_input_shape = x.shape
        self.cache = x

        out = self.pool(x, self.pool_size, self.strides, self.padding)

        if not self.channels_first:
            out = self.prot.transpose(out, perm=[0, 2, 3, 1])

        return out

    def backward(self, d_y: TFEVariable, learning_rate: float) -> Optional[TFEVariable]:
        raise NotImplementedError


class AveragePooling2D(Pooling2D):

    """
    AveragePooling2D

    :See: tf.nn.avg_pool
    """

    def __init__(self,
                 input_shape: List[int],
                 pool_size: IntTuple,
                 strides: Optional[IntTuple] = None,
                 padding: str = "SAME", channels_first: bool = True) -> None:

        super(AveragePooling2D, self).__init__(input_shape, pool_size, strides, padding, channels_first)

    def pool(self, x: TFEVariable, pool_size, strides, padding) -> TFEVariable:
        return self.prot.avgpool2d(x, pool_size, strides, padding)


class MaxPooling2D(Pooling2D):
    """
    MaxPooling2D

    :See: tf.nn.max_pool
    """

    def __init__(self,
                 input_shape: List[int],
                 pool_size: IntTuple,
                 strides: Optional[IntTuple] = None,
                 padding: str = "SAME", channels_first: bool = True) -> None:

        super(MaxPooling2D, self).__init__(input_shape, pool_size, strides, padding, channels_first)
        # TODO -- throw an error here if the protocol is not secureNN

    def pool(self, x: TFEVariable, pool_size, strides, padding) -> TFEVariable:
        return self.prot.maxpool2d(x, pool_size, strides, padding)
