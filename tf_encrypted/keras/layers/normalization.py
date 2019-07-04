"""Normalization layers implementation."""

import tensorflow as tf
from tensorflow.python.keras import initializers

from tf_encrypted.keras.engine import Layer
from tf_encrypted.keras.layers.layers_utils import default_args_check

class BatchNormalization(Layer):
  """Batch normalization layer (Ioffe and Szegedy, 2014).
  Normalize the activations of the previous layer at each batch,
  i.e. applies a transformation that maintains the mean activation
  close to 0 and the activation standard deviation close to 1.
  Arguments:
    axis: Integer, the axis that should be normalized
        (typically the features axis).
        For instance, after a `Conv2D` layer with
        `data_format="channels_first"`,
        set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
        If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
    fused: if `True`, use a faster, fused implementation, or raise a ValueError
      if the fused implementation cannot be used. If `None`, use the faster
      implementation if possible. If False, do not used the fused
      implementation.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    virtual_batch_size: An `int`. By default, `virtual_batch_size` is `None`,
      which means batch normalization is performed across the whole batch. When
      `virtual_batch_size` is not `None`, instead perform "Ghost Batch
      Normalization", which creates virtual sub-batches which are each
      normalized separately (with shared gamma, beta, and moving statistics).
      Must divide the actual batch size during execution.
    adjustment: A function taking the `Tensor` containing the (dynamic) shape of
      the input tensor and returning a pair (scale, bias) to apply to the
      normalized values (before gamma and beta), only during training. For
      example, if axis==-1,
        `adjustment = lambda shape: (
          tf.random_uniform(shape[-1:], 0.93, 1.07),
          tf.random_uniform(shape[-1:], -0.1, 0.1))`
      will scale the normalized value by up to 7% up or down, then shift the
      result by up to 0.1 (with independent scaling and bias for each feature
      but shared across all examples), and finally apply gamma and/or beta. If
      `None`, no adjustment is applied. Cannot be specified if
      virtual_batch_size is specified.
  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
  Output shape:
      Same shape as input.
  References:
      - [Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """
  def __init__(self,
               axis=3,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               fused=None, # pylint: disable=unused-argument
               trainable=False,
               virtual_batch_size=None,
               adjustment=None,
               name=None,
               **kwargs):
    super(BatchNormalization, self).__init__(
        name=name, trainable=trainable, **kwargs)

    self.beta_initializer = initializers.get(beta_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.moving_mean_initializer = initializers.get(moving_mean_initializer)
    self.moving_variance_initializer = initializers.get(
        moving_variance_initializer)

    default_args_check(beta_regularizer,
                       "beta_regularizer",
                       "BatchNormalization")
    default_args_check(gamma_regularizer,
                       "gamma_regularizer",
                       "BatchNormalization")
    default_args_check(beta_constraint,
                       "beta_constraint",
                       "BatchNormalization")
    default_args_check(gamma_constraint,
                       "gamma_constraint",
                       "BatchNormalization")
    default_args_check(renorm,
                       "renorm",
                       "BatchNormalization")
    default_args_check(renorm_clipping,
                       "renorm_clipping",
                       "BatchNormalization")
    default_args_check(virtual_batch_size,
                       "virtual_batch_size",
                       "BatchNormalization")
    default_args_check(adjustment,
                       "adjustment",
                       "BatchNormalization")

    # Axis -3 is equivalent to 1, and axis -1 is equivalent to 3, because the
    # input rank is required to be 4 (which is checked later).
    if axis not in (1, 3):
      raise ValueError("Axis of 1 or 3 is currently only supported")

    self.axis = axis
    self.scale = scale
    self.center = center
    self.epsilon = epsilon
    self.momentum = momentum
    self.renorm_momentum = renorm_momentum

  def build(self, input_shape):
    c = input_shape[self.axis]
    if len(input_shape) == 2:
      param_shape = [1, 1]
    elif len(input_shape) == 4:
      param_shape = [1, 1, 1, 1]

    param_shape[self.axis] = int(c)

    if self.scale:
      gamma = self.gamma_initializer(param_shape)
      self.gamma = self.prot.define_public_variable(gamma)
    else:
      self.gamma = None

    if self.center:
      beta = self.beta_initializer(param_shape)
      self.beta = self.prot.define_public_variable(beta)
    else:
      self.beta = None

    moving_mean = self.moving_mean_initializer(param_shape)
    self.moving_mean = self.prot.define_public_variable(moving_mean)

    moving_variance = self.moving_variance_initializer(param_shape)

    denomtemp = 1.0 / tf.sqrt(moving_variance + self.epsilon)
    self.denom = self.prot.define_public_variable(denomtemp)

    self.built = True

  def call(self, inputs):
    if self.beta is None and self.gamma is None:
      out = (inputs - self.moving_mean) * self.denom
    elif self.gamma is None:
      out = (inputs - self.moving_mean) * self.denom + self.beta
    elif self.beta is None:
      out = self.gamma * (inputs - self.moving_mean) * self.denom
    else:
      out = self.gamma * (inputs - self.moving_mean) * self.denom + self.beta
    return out

  def compute_output_shape(self, input_shape):
    return input_shape
