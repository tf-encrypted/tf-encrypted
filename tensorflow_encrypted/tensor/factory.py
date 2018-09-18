from abc import ABC, abstractmethod, abstractproperty
from typing import Type, List
from .tensor import AbstractTensor, AbstractConstant, AbstractVariable, AbstractPlaceholder


class AbstractFactory(ABC):
    @abstractproperty
    def Tensor(self) -> Type[AbstractTensor]:
        pass

    @abstractproperty
    def Constant(self) -> Type[AbstractConstant]:
        pass

    @abstractproperty
    def Variable(self) -> Type[AbstractVariable]:
        pass

    @abstractmethod
    def Placeholder(self, shape: List[int]) -> AbstractPlaceholder:
        pass

    @abstractproperty
    def modulus(self) -> int:
        pass

    @abstractmethod
    def stack(self, xs: List[AbstractTensor], axis: int=0) -> AbstractTensor:
        pass
