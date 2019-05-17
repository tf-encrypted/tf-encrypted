"""Activation Layer implementation."""
from tf_encrypted.keras.engine import Layer
from tf_encrypted.keras import activations

class Activation(Layer):
  """Applies an activation function to an output.
  Arguments:
      activation: name of activation function to use or
          TF Encrypted operation.
  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
  Output shape:
      Same shape as input.
  """

  def __init__(self, activation, **kwargs):
    super(Activation, self).__init__(**kwargs)
    self.activation = activations.get(activation)

  def build(self, input_shape):
    pass

  def call(self, inputs):
    return self.activation(inputs)

  def compute_output_shape(self, input_shape):
    return input_shape
