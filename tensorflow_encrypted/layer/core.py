import numpy as np
import math
from abc import ABC, abstractmethod
from ..protocol import get_protocol

# TODO
# split backward function in compute_gradient and compute_backpropagated_error?


class Layer(ABC):
    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    @property
    def prot(self):
        return get_protocol()
