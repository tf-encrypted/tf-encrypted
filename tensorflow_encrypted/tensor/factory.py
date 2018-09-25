from abc import ABC, abstractmethod
from typing import Type, List
from .tensor import AbstractTensor, AbstractConstant, AbstractVariable, AbstractPlaceholder


class AbstractFactory(ABC):

    @property
    @abstractmethod
    def Tensor(self) -> Type[AbstractTensor]:
        pass

    @property
    @abstractmethod
    def Constant(self) -> Type[AbstractConstant]:
        pass

    @property
    @abstractmethod
    def Variable(self) -> Type[AbstractVariable]:
        pass

    @abstractmethod
    def Placeholder(self, shape: List[int]) -> AbstractPlaceholder:
        pass

    @property
    @abstractmethod
    def modulus(self) -> int:
        pass
