"""Includes base classes used by all layer types."""

from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional

from tf_encrypted.protocol import TFEVariable

# TODO
# Split backward function in compute_gradient and compute_backpropagated_error?


class Layer(ABC):
    """
  Base class for all layers.
  """

    def __init__(self, input_shape: List[int]) -> None:
        self.input_shape = input_shape
        self.output_shape = self.get_output_shape()
        self.layer_output = None

    @abstractmethod
    def get_output_shape(self) -> List[int]:
        """Returns the layer's output shape"""

    @abstractmethod
    def initialize(self, *args, **kwargs) -> None:
        """Initialize any necessary tensors."""

    @abstractmethod
    def forward(self, *args, **kwargs) -> Optional[TFEVariable]:
        """Forward pass for inference"""

    # TODO[jason]: @abstractmethod
    def backward(self, *args, **kwargs) -> Optional[TFEVariable]:
        """Backward pass for training."""
