"""Convolutional Layer implementation."""
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import conv_utils

import tf_encrypted as tfe
from tf_encrypted.keras import activations
from tf_encrypted.keras import backend as KE
from tf_encrypted.keras.engine import Layer
from tf_encrypted.keras.layers.layers_utils import default_args_check
from tf_encrypted.protocol import TFEPrivateTensor

logger = logging.getLogger("tf_encrypted")


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

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):

        super(Conv2D, self).__init__(**kwargs)

        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, self.rank, "kernel_size"
        )
        if self.kernel_size[0] != self.kernel_size[1]:
            raise NotImplementedError(
                "TF Encrypted currently only supports same "
                "stride along the height and the width."
                "You gave: {}".format(self.kernel_size)
            )
        self.strides = conv_utils.normalize_tuple(strides, self.rank, "strides")
        self.padding = conv_utils.normalize_padding(padding).upper()
        self.data_format = conv_utils.normalize_data_format(data_format)
        if activation is not None:
            logger.info(
                "Performing an activation before a pooling layer can result "
                "in unnecessary performance loss. Check model definition in "
                "case of missed optimization."
            )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # Not implemented arguments
        default_args_check(dilation_rate, "dilation_rate", "Conv2D")
        default_args_check(kernel_regularizer, "kernel_regularizer", "Conv2D")
        default_args_check(bias_regularizer, "bias_regularizer", "Conv2D")
        default_args_check(activity_regularizer, "activity_regularizer", "Conv2D")
        default_args_check(kernel_constraint, "kernel_constraint", "Conv2D")
        default_args_check(bias_constraint, "bias_constraint", "Conv2D")

    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
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
        self._layer_input = inputs

        if self.data_format != "channels_first":
            inputs = tfe.transpose(inputs, perm=[0, 3, 1, 2])

        outputs = tfe.conv2d(inputs, self.kernel, self.strides[0], self.padding)

        if self.use_bias:
            outputs = outputs + self.bias

        if self.data_format != "channels_first":
            outputs = tfe.transpose(outputs, perm=[0, 2, 3, 1])

        if self.activation is not None:
            outputs = self.activation(outputs)

        self._layer_output = outputs
        return outputs

    def backward(self, d_y):
        x = self._layer_input
        y = self._layer_output
        kernel = self.weights[0]
        grad_weights = []

        # Convert to NCHW format
        if self.data_format != "channels_first":
            x = tfe.transpose(x, perm=[0, 3, 1, 2])
        n_x, _, h_x, w_x = x.shape.as_list()

        if self.activation is not None:
            self._activation_deriv = activations.get_deriv(self.activation.__name__)
            d_y = self._activation_deriv(y, d_y)

        # Convert to HWNC format
        if self.data_format == "channels_first":
            d_y = tfe.transpose(d_y, perm=[2, 3, 0, 1])
        else:
            d_y = tfe.transpose(d_y, perm=[1, 2, 0, 3])

        inner_padded_d_y = tfe.expand(d_y, self.strides[0])
        padded_d_y = tfe.pad(
            inner_padded_d_y,
            [
                [self.kernel_size[0] - 1, self.kernel_size[0] - 1],
                [self.kernel_size[1] - 1, self.kernel_size[1] - 1],
            ],
        )

        # Recover the NCHW format
        padded_d_y = tfe.transpose(padded_d_y, perm=[2, 3, 0, 1])

        # Flip h and w axis, and swap in and out channels
        flipped_kernel = tfe.transpose(tfe.reverse(kernel, [0, 1]), perm=[0, 1, 3, 2])

        # Back prop for dx:
        # https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710
        d_x = tfe.conv2d(padded_d_y, flipped_kernel, 1, "VALID")

        # Remove padding, if any
        if self.padding == "SAME":
            [[pad_top, pad_bottom], [pad_left, pad_right]] = self.pad_size(h_x, w_x)
            d_x = d_x[
                :,
                :,
                pad_top : (d_x.shape[2] - pad_bottom),
                pad_left : (d_x.shape[3] - pad_right),
            ]
            x = tfe.pad(
                x, [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]]
            )

        if self.data_format != "channels_first":
            d_x = tfe.transpose(d_x, perm=[0, 2, 3, 1])

        # Convert to CNHW
        x = tfe.transpose(x, perm=[1, 0, 2, 3])
        # Back prop for dw:
        # https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-fb2f2efc4faa
        # Output is in IOHW format
        d_kernel = tfe.conv2d(x, inner_padded_d_y, 1, "VALID")
        # Convert to HWIO
        d_kernel = tfe.transpose(d_kernel, perm=[2, 3, 0, 1])
        if self.lazy_normalization:
            d_kernel = d_kernel / n_x
        grad_weights.append(d_kernel)

        if self.use_bias:
            d_bias = d_y.reduce_sum(axis=[0, 1, 2]).reshape(self.bias.shape)
            if self.lazy_normalization:
                d_bias = d_bias / n_x
            grad_weights.append(d_bias)

        assert (
            d_x.shape == self._layer_input.shape
        ), "Different shapes: {} vs {}".format(d_x.shape, self._layer_input.shape)
        assert d_kernel.shape == self.kernel_shape, "Different shapes: {} vs {}".format(
            d_kernel.shape, self.kernel_shape
        )

        return grad_weights, d_x

    def pad_size(self, h_in, w_in):
        if h_in % self.strides[0] == 0:
            pad_along_height = max(self.kernel_size[0] - self.strides[0], 0)
        else:
            pad_along_height = max(self.kernel_size[0] - (h_in % self.strides[0]), 0)

        if w_in % self.strides[1] == 0:
            pad_along_width = max(self.kernel_size[1] - self.strides[1], 0)
        else:
            pad_along_width = max(self.kernel_size[1] - (w_in % self.strides[1]), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return [[pad_top, pad_bottom], [pad_left, pad_right]]

    def compute_output_shape(self, input_shape):
        """Compute output_shape for the layer."""
        h_filter, w_filter, _, n_filters = self.kernel_shape

        if self.data_format == "channels_first":
            n_x, _, h_x, w_x = input_shape.as_list()
        else:
            n_x, h_x, w_x, _ = input_shape.as_list()

        if self.padding == "SAME":
            h_out = int(np.ceil(float(h_x) / float(self.strides[0])))
            w_out = int(np.ceil(float(w_x) / float(self.strides[0])))
        if self.padding == "VALID":
            h_out = int(np.ceil(float(h_x - h_filter + 1) / float(self.strides[0])))
            w_out = int(np.ceil(float(w_x - w_filter + 1) / float(self.strides[0])))

        if self.data_format == "channels_first":
            return [n_x, n_filters, h_out, w_out]
        else:
            return [n_x, h_out, w_out, n_filters]


class DepthwiseConv2D(Conv2D):
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

    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        data_format=None,
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):

        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        self.rank = 2
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, self.rank, "kernel_size"
        )
        if self.kernel_size[0] != self.kernel_size[1]:
            raise NotImplementedError(
                "TF Encrypted currently only supports same "
                "stride along the height and the width."
                "You gave: {}".format(self.kernel_size)
            )
        self.strides = conv_utils.normalize_tuple(strides, self.rank, "strides")
        self.padding = conv_utils.normalize_padding(padding).upper()
        self.depth_multiplier = depth_multiplier
        self.data_format = conv_utils.normalize_data_format(data_format)
        if activation is not None:
            logger.info(
                "Performing an activation before a pooling layer can result "
                "in unnecessary performance loss. Check model definition in "
                "case of missed optimization."
            )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # Not implemented arguments
        default_args_check(
            depthwise_regularizer, "depthwise_regularizer", "DepthwiseConv2D",
        )
        default_args_check(
            bias_regularizer, "bias_regularizer", "DepthwiseConv2D",
        )
        default_args_check(
            activity_regularizer, "activity_regularizer", "DepthwiseConv2D",
        )
        default_args_check(
            depthwise_constraint, "depthwise_constraint", "DepthwiseConv2D",
        )
        default_args_check(
            bias_constraint, "bias_constraint", "DepthwiseConv2D",
        )

    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the inputs "
                "should be defined. Found `None`."
            )
        self.input_dim = int(input_shape[channel_axis])
        self.kernel_shape = self.kernel_size + (self.input_dim, self.depth_multiplier)

        kernel = self.depthwise_initializer(self.kernel_shape)
        kernel = self.rearrange_kernel(kernel)
        self.kernel = self.add_weight(kernel)

        if self.use_bias:
            # Expand bias shape dimensions. Bias needs to have
            # a rank of 3 to be added to the output
            bias_shape = [self.input_dim * self.depth_multiplier, 1, 1]
            bias = self.bias_initializer(bias_shape)
            self.bias = self.add_weight(bias)
        else:
            self.bias = None

        self.built = True

    def rearrange_kernel(self, kernel):
        """Rearrange kernel to match normal convolution kernels

    Arguments:
      kernel: kernel to be rearranged
    """
        mask = self.get_mask(self.input_dim)

        if isinstance(kernel, tf.Tensor):
            mask = tf.constant(
                mask.tolist(),
                dtype=tf.float32,
                shape=(
                    self.kernel_size[0],
                    self.kernel_size[1],
                    self.input_dim,
                    self.input_dim * self.depth_multiplier,
                ),
            )

            if self.depth_multiplier > 1:
                # rearrange kernel
                kernel = tf.reshape(
                    kernel,
                    shape=self.kernel_size
                    + (1, self.input_dim * self.depth_multiplier),
                )

            kernel = tf.multiply(kernel, mask)

        elif isinstance(kernel, np.ndarray):
            if self.depth_multiplier > 1:
                # rearrange kernel
                kernel = np.reshape(
                    kernel,
                    newshape=self.kernel_size
                    + (1, self.input_dim * self.depth_multiplier),
                )

            kernel = np.multiply(kernel, mask)

        elif isinstance(kernel, TFEPrivateTensor):
            mask = tfe.define_public_variable(mask)
            if self.depth_multiplier > 1:
                # rearrange kernel
                kernel = tfe.reshape(
                    kernel,
                    shape=self.kernel_size
                    + (1, self.input_dim * self.depth_multiplier),
                )

            kernel = tfe.mul(kernel, mask)
        else:
            raise ValueError("Invalid kernel type")

        return kernel

    def call(self, inputs):

        if self.data_format != "channels_first":
            inputs = tfe.transpose(inputs, perm=[0, 3, 1, 2])

        outputs = tfe.conv2d(inputs, self.kernel, self.strides[0], self.padding)

        if self.use_bias:
            outputs = outputs + self.bias

        if self.data_format != "channels_first":
            outputs = tfe.transpose(outputs, perm=[0, 2, 3, 1])

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def backward(self, d_y):
        raise NotImplementedError("Depthwise conv backward not implemented")

    def compute_output_shape(self, input_shape):
        """Compute output_shape for the layer."""
        h_filter, w_filter, _, n_filters = self.kernel_shape

        if self.data_format == "channels_first":
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
        mask = np.zeros(
            (
                self.kernel_size[0],
                self.kernel_size[1],
                in_channels,
                in_channels * self.depth_multiplier,
            )
        )
        for d in range(self.depth_multiplier):
            for i in range(in_channels):
                mask[:, :, i, d + (i * self.depth_multiplier)] = 1.0
        return mask

    def set_weights(self, weights, sess=None):
        """
    Sets the weights of the layer.

    Arguments:
      weights: A list of Numpy arrays with shapes and types
          matching the output of layer.get_weights() or a list
          of private variables
      sess: tfe session"""

        weights_types = (np.ndarray, TFEPrivateTensor)
        assert isinstance(weights[0], weights_types), type(weights[0])

        # Assign new keras weights to existing weights defined by
        # default when tfe layer was instantiated
        if not sess:
            sess = KE.get_session()

        if isinstance(weights[0], np.ndarray):
            for i, w in enumerate(self.weights):
                shape = w.shape.as_list()
                tfe_weights_pl = tfe.define_private_placeholder(shape)

                new_weight = weights[i]
                if i == 0:
                    # kernel
                    new_weight = self.rearrange_kernel(new_weight)
                else:
                    # bias
                    new_weight = new_weight.reshape(shape)

                fd = tfe_weights_pl.feed(new_weight)
                sess.run(tfe.assign(w, tfe_weights_pl), feed_dict=fd)

        elif isinstance(weights[0], TFEPrivateTensor):
            for i, w in enumerate(self.weights):
                shape = w.shape.as_list()

                new_weight = weights[i]
                if i == 0:
                    # kernel
                    new_weight = self.rearrange_kernel(new_weight)
                else:
                    # bias
                    new_weight = new_weight.reshape(shape)

                sess.run(tfe.assign(w, new_weight))
