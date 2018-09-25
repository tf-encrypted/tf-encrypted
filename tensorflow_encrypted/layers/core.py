from typing import Optional, List

from abc import ABC, abstractmethod
from ..protocol.protocol import get_protocol, Protocol
from ..protocol.pond import TFEVariable

# TODO
# Split backward function in compute_gradient and compute_backpropagated_error?


class Layer(ABC):

    def __init__(self, input_shape: List[int]) -> None:
        self.input_shape = input_shape
        self.output_shape = self.get_output_shape()
        self.layer_output = None

    @abstractmethod
    def get_output_shape(self) -> List[int]:
        """Returns the layer's output shape"""

    @abstractmethod
    def initialize(self, *args, **kwargs) -> None:  # type: ignore
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Optional[TFEVariable]:  # type: ignore
        pass

    @abstractmethod
    def backward(self, *args, **kwargs) -> Optional[TFEVariable]:  # type: ignore
        pass

    @property
    def prot(self) -> Optional[Protocol]:
        return get_protocol()
