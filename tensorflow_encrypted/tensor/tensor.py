from abc import ABC, abstractmethod
from typing import List, Union, Tuple

import numpy as np
import tensorflow as tf


class AbstractTensor(ABC):

    @abstractmethod
    def __init__(self, value: Union[np.ndarray, tf.Tensor]) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_native(cls, value: Union[np.ndarray, tf.Tensor]) -> 'AbstractTensor':
        assert isinstance(value, (np.ndarray, tf.Tensor)), type(value)
        return cls(value)

    @staticmethod
    @abstractmethod
    def stack(xs: List['AbstractTensor'], axis: int=0) -> 'AbstractTensor':
        pass

    @staticmethod
    @abstractmethod
    def concat(xs: List['AbstractTensor'], axis: int) -> 'AbstractTensor':
        pass

    @property
    @abstractmethod
    def shape(self) -> Union[Tuple[int, ...], tf.TensorShape]:
        pass

    @classmethod
    def lift(cls, x: Union['AbstractTensor', int]) -> 'AbstractTensor':
        # TODO[Morten] support other types of `x`

        if isinstance(x, AbstractTensor):
            return x

        if type(x) is int:
            return cls.from_native(np.array([x]))

        raise TypeError("Unsupported type {}".format(type(x)))


class AbstractConstant(AbstractTensor):
    pass


class AbstractPlaceholder(AbstractTensor):
    pass


class AbstractVariable(AbstractTensor):
    pass
