import numpy as np
import math
from types import NoneType
from typing import Optional
from abc import ABC, abstractmethod
from ..protocol import get_protocol, Protocol

# TODO
# split backward function in compute_gradient and compute_backpropagated_error?


class Layer(ABC):
    @abstractmethod
    def initialize(self, *args, **kwargs) -> NoneType:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    @property
    def prot(self) -> Optional[Protocol]:
        return get_protocol()
