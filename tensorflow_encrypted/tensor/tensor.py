from abc import ABC, abstractmethod
from typing import List


class AbstractTensor(ABC):
    @staticmethod
    @abstractmethod
    def stack(xs: List['AbstractTensor'], axis: int=0) -> 'AbstractTensor':
        pass

    @staticmethod
    @abstractmethod
    def concat(xs: List['AbstractTensor'], axis: int) -> 'AbstractTensor':
        pass


class AbstractConstant(AbstractTensor):
    pass


class AbstractPlaceholder(AbstractTensor):
    pass


class AbstractVariable(AbstractTensor):
    pass
