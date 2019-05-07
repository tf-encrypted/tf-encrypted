# pylint: disable=arguments-differ
"""Reshape Layer object."""
from typing import List
import numpy as np

from .core import Layer


class Reshape(Layer):
  """
  Reshape Layer

  :See: tf.keras.layers.Reshape
  """

  def __init__(self, input_shape, output_shape=None) -> None:
    if output_shape is None:
      self.output_shape = [-1]
    self.output_shape = output_shape

    super(Reshape, self).__init__(input_shape)

  def get_output_shape(self) -> List[int]:
    """Returns the layer's output shape"""
    if -1 not in self.output_shape:
      return self.output_shape

    total_input_dims = np.prod(self.input_shape)

    dim = 1
    for i in self.output_shape:
      if i != -1:
        dim *= i
    missing_dim = int(total_input_dims / dim)

    output_shape = self.output_shape
    for ix, dim in enumerate(output_shape):
      if dim == -1:
        output_shape[ix] = missing_dim

    return output_shape

  def initialize(self, *args, **kwargs) -> None:
    pass

  def forward(self, x):
    y = self.prot.reshape(x, self.output_shape)
    self.layer_output = y
    return y

  def backward(self, *args, **kwargs):
    raise NotImplementedError()
