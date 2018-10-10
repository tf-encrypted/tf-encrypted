import abc
from typing import List, Tuple, Union, TypeVar, Generic

import tensorflow as tf


class AbstractTensor(abc.ABC):

    @property
    @abc.abstractmethod
    def factory(self) -> 'AbstractFactory':
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> Union[Tuple[int, ...], tf.TensorShape]:
        pass


class AbstractConstant(AbstractTensor):
    pass


class AbstractPlaceholder(AbstractTensor):
    pass


class AbstractVariable(AbstractTensor):
    pass


T = TypeVar('T')
C = TypeVar('C')
V = TypeVar('V')
P = TypeVar('P')


class AbstractFactory(abc.ABC, Generic[T, C, V, P]):

    @abc.abstractmethod
    def tensor(self, value) -> T:
        pass

    @abc.abstractmethod
    def constant(self, value) -> C:
        pass

    @abc.abstractmethod
    def variable(self, initial_value) -> V:
        pass

    @abc.abstractmethod
    def placeholder(self, shape: List[int]) -> P:
        pass

    @abc.abstractmethod
    def sample_uniform(self, shape: List[int]) -> T:
        pass

    @abc.abstractmethod
    def stack(self, xs: List[T], axis: int = 0) -> T:
        pass

    @abc.abstractmethod
    def concat(self, xs: List[T], axis: int) -> T:
        pass

    @property
    @abc.abstractmethod
    def modulus(self) -> int:
        pass
