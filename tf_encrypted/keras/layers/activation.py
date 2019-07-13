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
    self.activation_deriv = activations.get_deriv(activation)

  def build(self, input_shape):
    pass

  def call(self, inputs):
    y = self.activation(inputs)
    self._layer_output = y
    return y

  def compute_output_shape(self, input_shape):
    return input_shape

  def backward(self, d_y):
    y = self._layer_output
    grad_weights = []
    d_x = d_y * self.activation_deriv(y)
    return grad_weights, d_x
