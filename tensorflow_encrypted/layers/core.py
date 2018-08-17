import numpy as np
import math
from typing import Any, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import tensorflow as tf
from ..protocol.protocol import get_protocol, Protocol
from ..protocol.types import TFEVariable

# TODO
# split backward function in compute_gradient and compute_backpropagated_error?


class Layer(ABC):
    @abstractmethod
    def initialize(self, *args, **kwargs) -> None:  # type: ignore
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> TFEVariable:  # type: ignore
        pass

    @abstractmethod
    def backward(self, *args, **kwargs) -> Optional[TFEVariable]:  # type: ignore
        pass

    @property
    def prot(self) -> Optional[Protocol]:
        return get_protocol()
