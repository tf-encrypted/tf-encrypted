"""Includes base classes used by all layer types."""

from typing import List, Optional
from abc import ABC, abstractmethod

from tf_encrypted.protocol.pond import TFEVariable
from tf_encrypted.protocol.protocol import get_protocol, Protocol

# TODO
# Split backward function in compute_gradient and compute_backpropagated_error?


class Layer(ABC):
  """
  Base layer class.
  This is the class from which all layers inherit.
  A layer is a class implementing common neural networks operations, such
  as convolution, batch norm, etc. These operations require managing weights,
  losses, updates, and inter-layer connectivity.
  Users will just instantiate a layer and then treat it as a callable.
  We recommend that descendants of `Layer` implement the following methods:
  * `__init__()`: Save configuration in member variables
  * `build()`: Called once from `__call__`, when we know the shapes of inputs
    and `dtype`. Should have the calls to `add_weight()`, and then
    call the super's `build()` (which sets `self.built = True`, which is
    nice in case the user wants to call `build()` manually before the
    first `__call__`).
  * `call()`: Called in `__call__` after making sure `build()` has been called
    once. Should actually perform the logic of applying the layer to the
    input tensors (which should be passed in as the first argument).
  """

  def __init__(self, trainable=True, **kwargs) -> None:

    self.trainable = trainable
    self.built = False

  @abstractmethod
  def build(self, input_shape) -> None:
    """Creates the variables of the layer (optional, for subclass implementers).
    This is a method that implementers of subclasses of `Layer` or `Model`
    can override if they need a state-creation step in-between
    layer instantiation and layer call.
    This is typically used to create the weights of `Layer` subclasses.
    Arguments:
      input_shape: Instance of `TensorShape`, or list of instances of
        `TensorShape` if the layer expects a list of inputs
        (one instance per input).
    """
    self.built = True

  @abstractmethod
  def call(self, inputs, **kwargs) -> Optional[TFEVariable]:
    """This is where the layer's logic lives.
    Arguments:
        inputs: Input tensor, or list/tuple of input tensors.
        **kwargs: Additional keyword arguments.
    Returns:
        A tensor or list/tuple of tensors.
    """
    return inputs

  def add_weight(self, *args, **kargs):
    pass

  @abstractmethod
  def compute_output_shape(self, input_shape) -> List[int]:
    """Returns the layer's output shape"""

  def __call__(self, inputs, *args, **kargs):
    """Wraps `call`, applying pre- and post-processing steps.
    Arguments:
      inputs: input tensor(s).
      *args: additional positional arguments to be passed to `self.call`.
      **kwargs: additional keyword arguments to be passed to `self.call`.
    Returns:
      Output tensor(s).
    """
    if not self.built:
      self._maybe_build(inputs)
      self.built = True

    outputs = self.call(inputs, *args, **kargs)

    return outputs

  def _maybe_build(self, inputs):

    input_shapes = inputs.shape

    self.build(input_shapes)

  # TODO[jason]: @abstractmethod
  def backward(self, *args, **kwargs) -> Optional[TFEVariable]:
    """Backward pass for training."""

  @property
  def prot(self) -> Optional[Protocol]:
    return get_protocol()
