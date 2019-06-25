"""Activation Layer implementation."""
from tf_encrypted.keras.engine import Layer
from tf_encrypted.keras.activations import relu
from tf_encrypted.keras.layers.layers_utils import default_args_check


class ReLU(Layer):
  """Rectified Linear Unit activation function.
  With default values, it returns element-wise `max(x, 0)`.
  Otherwise, it follows:
  `f(x) = max_value` for `x >= max_value`,
  `f(x) = x` for `threshold <= x < max_value`,
  `f(x) = negative_slope * (x - threshold)` otherwise.
  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
  Output shape:
      Same shape as the input.
  Arguments:
      max_value: float >= 0. Maximum activation value.
      negative_slope: float >= 0. Negative slope coefficient.
      threshold: float. Threshold value for thresholded activation.
  """

  def __init__(self, max_value=None, negative_slope=0, threshold=0, **kwargs):
    super(ReLU, self).__init__(**kwargs)

    # Not implemented arguments
    default_args_check(max_value, "max_value", "ReLU")
    default_args_check(negative_slope, "negative_slope", "ReLU")
    default_args_check(threshold, "threshold", "ReLU")

  def build(self, input_shape):
    self.built = True

  def call(self, inputs):
    return relu(inputs)

  def compute_output_shape(self, input_shape):
    return input_shape
