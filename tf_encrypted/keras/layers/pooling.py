"""Pooling Layer implementation."""
import math
from abc import abstractmethod

from tensorflow.python.keras.utils import conv_utils

import tf_encrypted as tfe
from tf_encrypted.keras.engine import Layer


class Pooling2D(Layer):
    """Pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).
  This class only exists for code reuse. It will never be an exposed API.
  Arguments:
    _pool_function: The pooling function to apply, e.g. `prot.max_pool2d`.
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
  """

    def __init__(
        self,
        _pool_function,
        pool_size,
        strides,
        padding="valid",
        data_format=None,
        **kwargs,
    ):
        super(Pooling2D, self).__init__(**kwargs)

        if data_format is None:
            data_format = "channels_last"
        if strides is None:
            strides = pool_size
        self._pool_function = _pool_function
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, "pool_size")
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        self.padding = conv_utils.normalize_padding(padding).upper()
        self.data_format = conv_utils.normalize_data_format(data_format)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):

        if self.data_format != "channels_first":
            inputs = tfe.transpose(inputs, perm=[0, 3, 1, 2])

        outputs = self._pool_function(
            inputs, self.pool_size, self.strides, self.padding
        )

        if self.data_format != "channels_first":
            outputs = tfe.transpose(outputs, perm=[0, 2, 3, 1])

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            _, _, h_in, w_in = input_shape
        else:
            _, h_in, w_in, _ = input_shape

        if self.padding == "SAME":
            h_out = math.ceil(h_in / self.strides[0])
            w_out = math.ceil(w_in / self.strides[1])
        else:
            h_out = math.ceil((h_in - self.pool_size[0] + 1) / self.strides[0])
            w_out = math.ceil((w_in - self.pool_size[1] + 1) / self.strides[1])

        if self.data_format == "channels_first":
            return [input_shape[0], input_shape[1], h_out, w_out]
        else:
            return [input_shape[0], h_out, w_out, input_shape[3]]


class MaxPooling2D(Pooling2D):
    """Max pooling operation for spatial data.
  Arguments:
      pool_size: integer or tuple of 2 integers,
          factors by which to downscale (vertical, horizontal).
          (2, 2) will halve the input in both spatial dimension.
          If only one integer is specified, the same window length
          will be used for both dimensions.
      strides: Integer, tuple of 2 integers, or None.
          Strides values.
          If None, it will default to `pool_size`.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
  Input shape:
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, rows, cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, rows, cols)`
  Output shape:
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, pooled_rows, pooled_cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, pooled_rows, pooled_cols)`
  """

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs,
    ):
        super(MaxPooling2D, self).__init__(
            tfe.maxpool2d,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs,
        )

    def call(self, inputs):
        self._layer_input = inputs

        if self.data_format != "channels_first":
            inputs = tfe.transpose(inputs, perm=[0, 3, 1, 2])

        outputs, outputs_argmax = tfe.maxpool2d_with_argmax(
            inputs, self.pool_size, self.strides, self.padding
        )

        if self.data_format != "channels_first":
            outputs = tfe.transpose(outputs, perm=[0, 2, 3, 1])
            outputs_argmax = tfe.transpose(outputs_argmax, perm=[0, 2, 3, 1, 4])

        self._layer_output = (outputs, outputs_argmax)

        return outputs

    def backward(self, d_y):
        x = self._layer_input
        y, y_arg = self._layer_output

        if self.data_format != "channels_first":
            d_y = tfe.transpose(d_y, perm=[0, 3, 1, 2])
            y_arg = tfe.transpose(y_arg, perm=[0, 3, 1, 2, 4])
            h_x, w_x = (x.shape[1], x.shape[2])
        else:
            h_x, w_x = (x.shape[2], x.shape[3])

        batch, channels, h_y, w_y, pool_len = y_arg.shape

        _d_y = tfe.tile(tfe.expand_dims(d_y, axis=4), [1, 1, 1, 1, pool_len])
        _d_y = _d_y * y_arg
        _d_y = tfe.reshape(_d_y, [batch * channels, h_y, w_y, pool_len])

        d_x = tfe.patches2im(
            _d_y,
            self.pool_size,
            stride=self.strides[0],
            padding=self.padding,
            img_size=(h_x, w_x),
            consolidation="SUM",
            data_format="NHWC",
        )
        d_x = tfe.reshape(d_x, [batch, channels, h_x, w_x])

        if self.data_format != "channels_first":
            d_x = tfe.transpose(d_x, perm=[0, 2, 3, 1])

        return [], d_x


class AveragePooling2D(Pooling2D):
    """Average pooling operation for spatial data.
  Arguments:
      pool_size: integer or tuple of 2 integers,
          factors by which to downscale (vertical, horizontal).
          (2, 2) will halve the input in both spatial dimension.
          If only one integer is specified, the same window length
          will be used for both dimensions.
      strides: Integer, tuple of 2 integers, or None.
          Strides values.
          If None, it will default to `pool_size`.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
  Input shape:
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, rows, cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, rows, cols)`
  Output shape:
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, pooled_rows, pooled_cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, pooled_rows, pooled_cols)`
  """

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs,
    ):
        super(AveragePooling2D, self).__init__(
            tfe.avgpool2d,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs,
        )

    def call(self, inputs):
        self._layer_input = inputs

        if self.data_format != "channels_first":
            inputs = tfe.transpose(inputs, perm=[0, 3, 1, 2])

        outputs = tfe.avgpool2d(inputs, self.pool_size, self.strides, self.padding)

        if self.data_format != "channels_first":
            outputs = tfe.transpose(outputs, perm=[0, 2, 3, 1])

        self._layer_output = outputs

        return outputs

    def backward(self, d_y):
        x = self._layer_input
        pool_len = self.pool_size[0] * self.pool_size[1]
        scalar = 1 / pool_len

        if self.data_format != "channels_first":
            d_y = tfe.transpose(d_y, perm=[0, 3, 1, 2])
            h_x, w_x = (x.shape[1], x.shape[2])
        else:
            h_x, w_x = (x.shape[2], x.shape[3])

        batch, channels, h_y, w_y = d_y.shape

        _d_y = tfe.tile(tfe.expand_dims(d_y, 4), [1, 1, 1, 1, pool_len])
        _d_y = _d_y * scalar
        _d_y = tfe.reshape(_d_y, [batch * channels, h_y, w_y, pool_len])

        d_x = tfe.patches2im(
            _d_y,
            self.pool_size,
            stride=self.strides[0],
            padding=self.padding,
            img_size=(h_x, w_x),
            consolidation="SUM",
            data_format="NHWC",
        )
        d_x = tfe.reshape(d_x, [batch, channels, h_x, w_x])

        if self.data_format != "channels_first":
            d_x = tfe.transpose(d_x, perm=[0, 2, 3, 1])

        return [], d_x


class GlobalPooling2D(Layer):
    """Abstract class for different global pooling 2D layers.
  """

    def __init__(self, data_format=None, **kwargs):
        super(GlobalPooling2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
            output_shape = [input_shape[0], input_shape[3]]
        else:
            output_shape = [input_shape[0], input_shape[1]]

        return output_shape

    @abstractmethod
    def call(self, inputs):
        raise NotImplementedError


class GlobalAveragePooling2D(GlobalPooling2D):
    """Global average pooling operation for spatial data.

  Arguments:
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

  Input shape:
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, rows, cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, rows, cols)`

  Output shape:
      2D tensor with shape:
      `(batch_size, channels)`
  """

    def build(self, input_shape):
        if self.data_format == "channels_last":
            _, h_in, w_in, _ = input_shape
        else:
            _, _, h_in, w_in = input_shape

        self.scalar = 1 / int(h_in * w_in)

    def call(self, inputs):
        self._layer_input = inputs
        if self.data_format == "channels_last":
            x_reduced = inputs.reduce_sum(axis=2).reduce_sum(axis=1)
        else:
            x_reduced = inputs.reduce_sum(axis=3).reduce_sum(axis=2)

        return x_reduced * self.scalar

    def backward(self, d_y):
        x = self._layer_input
        if self.data_format == "channels_last":
            h_x, w_x = (x.shape[1], x.shape[2])
        else:
            h_x, w_x = (x.shape[2], x.shape[3])

        _d_y = d_y * self.scalar
        _d_y = tfe.expand_dims(tfe.expand_dims(_d_y, axis=2), axis=3)

        d_x = tfe.tile(_d_y, multiples=[1, 1, h_x, w_x])

        if self.data_format == "channels_last":
            d_x = tfe.transpose(d_x, perm=[0, 2, 3, 1])

        return [], d_x


class GlobalMaxPooling2D(GlobalPooling2D):
    """Global max pooling operation for spatial data.

  Arguments:
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

  Input shape:
      - If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, rows, cols, channels)`
      - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, rows, cols)`

  Output shape:
      2D tensor with shape:
      `(batch_size, channels)`
  """

    def call(self, inputs):
        self._layer_input = inputs

        if self.data_format == "channels_last":
            inputs = tfe.transpose(inputs, perm=[0, 3, 1, 2])
        batch, channels, h_x, w_x = inputs.shape
        inputs = tfe.reshape(inputs, [batch, channels, -1])

        outputs, outputs_argmax = tfe.reduce_max_with_argmax(inputs, axis=2)

        self._layer_output = (outputs, outputs_argmax)

        return outputs

    def backward(self, d_y):
        x = self._layer_input
        y, y_arg = self._layer_output

        if self.data_format == "channels_last":
            batch, h_x, w_x, channels = x.shape
        else:
            batch, channels, h_x, w_x = x.shape

        _d_y = tfe.expand_dims(d_y, axis=2)
        d_x = tfe.tile(_d_y, multiples=[1, 1, h_x * w_x])
        d_x = d_x * y_arg
        d_x = tfe.reshape(d_x, [batch, channels, h_x, w_x])

        if self.data_format == "channels_last":
            d_x = tfe.transpose(d_x, perm=[0, 2, 3, 1])

        return [], d_x
