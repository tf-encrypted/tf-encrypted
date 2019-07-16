"""Convolutional Layer implementation."""
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import conv_utils

from tf_encrypted.keras.engine import Layer
from tf_encrypted.keras import activations
from tf_encrypted.keras.layers.layers_utils import default_args_check

logger = logging.getLogger('tf_encrypted')

class Conv2D(Layer):
  """2D convolution layer (e.g. spatial convolution over images).
  This layer creates a convolution kernel that is convolved
  with the layer input to produce a tensor of
  outputs. If `use_bias` is True,
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.
  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`.
  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
          height and width of the 2D convolution window.
          Can be a single integer to specify the same value for
          all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the height and width.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      dilation_rate: an integer or tuple/list of 2 integers, specifying
          the dilation rate to use for dilated convolution.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any stride value != 1.
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
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
  Input shape:
      4D tensor with shape:
      `(samples, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, rows, cols, channels)` if data_format='channels_last'.
  Output shape:
      4D tensor with shape:
      `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
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

    super(Conv2D, self).__init__(**kwargs)

    self.rank = 2
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, self.rank, 'kernel_size')
    if self.kernel_size[0] != self.kernel_size[1]:
      raise NotImplementedError("TF Encrypted currently only supports same "
                                "stride along the height and the width."
                                "You gave: {}".format(self.kernel_size))
    self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding).upper()
    self.data_format = conv_utils.normalize_data_format(data_format)
    if activation is not None:
      logger.info("Performing an activation before a pooling layer can result "
                  "in unnecessary performance loss. Check model definition in "
                  "case of missed optimization.")
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    # Not implemented arguments
    default_args_check(dilation_rate, "dilation_rate", "Conv2D")
    default_args_check(kernel_regularizer, "kernel_regularizer", "Conv2D")
    default_args_check(bias_regularizer, "bias_regularizer", "Conv2D")
    default_args_check(activity_regularizer,
                       "activity_regularizer",
                       "Conv2D")
    default_args_check(kernel_constraint, "kernel_constraint", "Conv2D")
    default_args_check(bias_constraint, "bias_constraint", "Conv2D")

  def build(self, input_shape):
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.kernel_shape = self.kernel_size + (input_dim, self.filters)

    kernel = self.kernel_initializer(self.kernel_shape)
    self.kernel = self.add_weight(kernel)

    if self.use_bias:
      # Expand bias shape dimensions. Bias needs to have
      # a rank of 3 to be added to the output
      bias_shape = [self.filters, 1, 1]
      bias = self.bias_initializer(bias_shape)
      self.bias = self.add_weight(bias)
    else:
      self.bias = None

    self.built = True

  def call(self, inputs):

    if self.data_format != 'channels_first':
      inputs = self.prot.transpose(inputs, perm=[0, 3, 1, 2])

    outputs = self.prot.conv2d(inputs,
                               self.kernel,
                               self.strides[0],
                               self.padding)

    if self.use_bias:
      outputs = outputs + self.bias

    if self.data_format != 'channels_first':
      outputs = self.prot.transpose(outputs, perm=[0, 2, 3, 1])

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    """Compute output_shape for the layer."""
    h_filter, w_filter, _, n_filters = self.kernel_shape

    if self.data_format == 'channels_first':
      n_x, _, h_x, w_x = input_shape.as_list()
    else:
      n_x, h_x, w_x, _ = input_shape.as_list()

    if self.padding == "SAME":
      h_out = int(np.ceil(float(h_x) / float(self.strides[0])))
      w_out = int(np.ceil(float(w_x) / float(self.strides[0])))
    if self.padding == "VALID":
      h_out = int(np.ceil(float(h_x - h_filter + 1) / float(self.strides[0])))
      w_out = int(np.ceil(float(w_x - w_filter + 1) / float(self.strides[0])))

    return [n_x, n_filters, h_out, w_out]


class DepthwiseConv2D(Layer):
  """Depthwise separable 2D convolution.

  Depthwise Separable convolutions consists in performing
  just the first step in a depthwise spatial convolution
  (which acts on each input channel separately).
  The `depth_multiplier` argument controls how many
  output channels are generated per input channel in the depthwise step.

  Arguments:
    kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    padding: one of `'valid'` or `'same'` (case-insensitive).
    depth_multiplier: The number of depthwise convolution output channels
        for each input channel.
        The total number of depthwise convolution output
        channels will be equal to `filters_in * depth_multiplier`.
    data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be 'channels_last'.
    activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. 'linear' activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    depthwise_initializer: Initializer for the depthwise kernel matrix.
    bias_initializer: Initializer for the bias vector.
    depthwise_regularizer: Regularizer function applied to
        the depthwise kernel matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
        the output of the layer (its 'activation').
    depthwise_constraint: Constraint function applied to
        the depthwise kernel matrix.
    bias_constraint: Constraint function applied to the bias vector.

  Input shape:
    4D tensor with shape:
    `[batch, channels, rows, cols]` if data_format='channels_first'
    or 4D tensor with shape:
    `[batch, rows, cols, channels]` if data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `[batch, filters, new_rows, new_cols]` if data_format='channels_first'
    or 4D tensor with shape:
    `[batch, new_rows, new_cols, filters]` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.
  """

  def __init__(self,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               **kwargs):

    super(DepthwiseConv2D, self).__init__(**kwargs)

    self.rank = 2
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, self.rank, 'kernel_size')
    if self.kernel_size[0] != self.kernel_size[1]:
      raise NotImplementedError("TF Encrypted currently only supports same "
                                "stride along the height and the width."
                                "You gave: {}".format(self.kernel_size))
    self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding).upper()
    self.depth_multiplier = depth_multiplier
    self.data_format = conv_utils.normalize_data_format(data_format)
    if activation is not None:
      logger.info("Performing an activation before a pooling layer can result "
                  "in unnecessary performance loss. Check model definition in "
                  "case of missed optimization.")
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.depthwise_initializer = initializers.get(depthwise_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    # Not implemented arguments
    default_args_check(depthwise_regularizer,
                       "depthwise_regularizer",
                       "DepthwiseConv2D")
    default_args_check(bias_regularizer,
                       "bias_regularizer",
                       "DepthwiseConv2D")
    default_args_check(activity_regularizer,
                       "activity_regularizer",
                       "DepthwiseConv2D")
    default_args_check(depthwise_constraint,
                       "depthwise_constraint",
                       "DepthwiseConv2D")
    default_args_check(bias_constraint,
                       "bias_constraint",
                       "DepthwiseConv2D")

  def build(self, input_shape):
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.kernel_shape = self.kernel_size + (input_dim, self.depth_multiplier)

    kernel = self.depthwise_initializer(self.kernel_shape)
    mask = tf.constant(self.get_mask(input_dim).tolist(),
                       dtype=tf.float32,
                       shape=(self.kernel_size[0],
                              self.kernel_size[1],
                              input_dim * self.depth_multiplier,
                              input_dim))

    if self.depth_multiplier > 1:
      # re-arange kernel
      kernel = tf.transpose(kernel, [0, 1, 3, 2])
      kernel = tf.reshape(kernel, shape=self.kernel_size +
                          (input_dim * self.depth_multiplier, 1))

    kernel = tf.multiply(kernel, mask)
    self.kernel = self.add_weight(kernel)

    if self.use_bias:
      # Expand bias shape dimensions. Bias needs to have
      # a rank of 3 to be added to the output
      bias_shape = [input_dim * self.depth_multiplier, 1, 1]
      bias = self.bias_initializer(bias_shape)
      self.bias = self.add_weight(bias)
    else:
      self.bias = None

    self.built = True

  def call(self, inputs):

    if self.data_format != 'channels_first':
      inputs = self.prot.transpose(inputs, perm=[0, 3, 1, 2])

    outputs = self.prot.conv2d(inputs,
                               self.kernel,
                               self.strides[0],
                               self.padding)

    if self.use_bias:
      outputs = outputs + self.bias

    if self.data_format != 'channels_first':
      outputs = self.prot.transpose(outputs, perm=[0, 2, 3, 1])

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    """Compute output_shape for the layer."""
    h_filter, w_filter, _, n_filters = self.kernel_shape

    if self.data_format == 'channels_first':
      n_x, _, h_x, w_x = input_shape.as_list()
    else:
      n_x, h_x, w_x, _ = input_shape.as_list()

    if self.padding == "SAME":
      h_out = int(np.ceil(float(h_x) / float(self.strides[0])))
      w_out = int(np.ceil(float(w_x) / float(self.strides[0])))
    if self.padding == "VALID":
      h_out = int(np.ceil(float(h_x - h_filter + 1) / float(self.strides[0])))
      w_out = int(np.ceil(float(w_x - w_filter + 1) / float(self.strides[0])))

    return [n_x, n_filters, h_out, w_out]

  def get_mask(self, in_channels):
    mask = np.zeros((self.kernel_size[0],
                     self.kernel_size[1],
                     in_channels, in_channels * self.depth_multiplier))
    for d in range(self.depth_multiplier):
      for i in range(in_channels):
        mask[:, :, i, i + (d * in_channels)] = 1.
    return np.transpose(mask, [0, 1, 3, 2])
