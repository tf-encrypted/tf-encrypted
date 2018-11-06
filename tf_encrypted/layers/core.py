from typing import Optional, List, Any, Union
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from ..protocol.protocol import get_protocol, Protocol
from ..protocol.pond import PondPublicTensor, PondPrivateTensor, TFEVariable

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
    def initialize(
        self,
        *args: Optional[Union[np.ndarray, tf.Tensor, PondPublicTensor, PondPrivateTensor]],
        **kwargs: Optional[Union[np.ndarray, tf.Tensor, PondPublicTensor, PondPrivateTensor]]
    ) -> None:
        pass

    @abstractmethod
    def forward(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Optional[TFEVariable]:
        """
        Forward pass for inference
        """
        pass

    @abstractmethod
    def backward(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Optional[TFEVariable]:
        """
        The backward pass for training.
        """
        pass

    @property
    def prot(self) -> Optional[Protocol]:
        return get_protocol()
