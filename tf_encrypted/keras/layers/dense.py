# pylint: disable=arguments-differ
"""Dense (i.e. fully connected) Layer implementation."""
from tensorflow.python.keras import initializers

from tf_encrypted.keras.engine import Layer
from tf_encrypted.keras import activations
from tf_encrypted.keras.layers.layers_utils import default_args_check

class Dense(Layer):
  """Just your regular densely-connected NN layer.
  `Dense` implements the operation:
  `output = activation(dot(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights matrix
  created by the layer, and `bias` is a bias vector created by the layer
  (only applicable if `use_bias` is `True`).

  Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation").
      kernel_constraint: Constraint function applied to
          the `kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.

  Input shape:
      2D tensor with shape: `(batch_size, input_dim)`.

  Output shape:
      2D tensor with shape: `(batch_size, units)`.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):

    super(Dense, self).__init__(**kwargs)

    self.units = int(units)
    self.activation = activations.get(activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    # Not implemented arguments
    default_args_check(kernel_regularizer, "kernel_regularizer", "Dense")
    default_args_check(bias_regularizer, "bias_regularizer", "Dense")
    default_args_check(activity_regularizer,
                       "activity_regularizer",
                       "Dense")
    default_args_check(kernel_constraint, "kernel_constraint", "Dense")
    default_args_check(bias_constraint, "bias_constraint", "Dense")

  def compute_output_shape(self, input_shape):
    return [input_shape[0], self.units]

  def build(self, input_shape):

    rank = len(input_shape)

    if rank > 2:
      raise NotImplementedError(
          "For dense layer, TF Encrypted currently support only input with "
          "a rank equal to 2 instead of {}.".format(len(input_shape)))

    units_in = int(input_shape[1])
    kernel = self.kernel_initializer([units_in,
                                      self.units])
    self.kernel = self.add_weight(kernel)

    if self.use_bias:
      bias = self.bias_initializer([self.units])
      self.bias = self.add_weight(bias)
    else:
      self.bias = None

    self.built = True

  def call(self, inputs):

    if self.use_bias:
      outputs = inputs.matmul(self.kernel) + self.bias
    else:
      outputs = inputs.matmul(self.kernel)

    if self.activation is not None:
      return self.activation(outputs)

    return outputs
