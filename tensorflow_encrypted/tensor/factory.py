import abc
from typing import List, Type, Union

import numpy as np
import tensorflow as tf


class AbstractFactory(abc.ABC):

    @abc.abstractmethod
    def tensor(self, value) -> Type['AbstractTensor']:
        pass

    @abc.abstractmethod
    def constant(self, value) -> Type['AbstractConstant']:
        pass

    @abc.abstractmethod
    def variable(self, initial_value) -> Type['AbstractVariable']:
        pass

    @abc.abstractmethod
    def placeholder(self, shape: List[int]) -> Type['AbstractPlaceholder']:
        pass

    @property
    @abc.abstractmethod
    def modulus(self) -> int:
        pass


class AbstractTensor(abc.ABC):

    @property
    @abc.abstractmethod
    def factory(self) -> AbstractFactory:
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> List[int]:
        pass


class AbstractConstant(AbstractTensor):
    pass


class AbstractPlaceholder(AbstractTensor):
    pass


class AbstractVariable(AbstractTensor):
    pass
