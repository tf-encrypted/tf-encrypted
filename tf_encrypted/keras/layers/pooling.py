"""Pooling Layer implementation."""
from tensorflow.python.keras.utils import conv_utils

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

  def __init__(self, _pool_function, pool_size, strides,
               padding='valid', data_format=None,
               **kwargs):
    super(Pooling2D, self).__init__(**kwargs)

    if data_format is None:
      data_format = 'channels_last'
    if strides is None:
      strides = pool_size
    self._pool_function = _pool_function
    self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding).upper()
    self.data_format = conv_utils.normalize_data_format(data_format)

  def build(self, input_shape):
    self.built = True

  def call(self, inputs):

    if self.data_format != 'channels_first':
      inputs = self.prot.transpose(inputs, perm=[0, 3, 1, 2])

    outputs = self._pool_function(inputs,
                                  self.pool_size,
                                  self.strides,
                                  self.padding)

    if self.data_format != 'channels_first':
      outputs = self.prot.transpose(outputs, perm=[0, 2, 3, 1])

    return outputs

  def compute_output_shape(self, input_shape):
    if self.channels_first:
      _, _, h_in, w_in = input_shape
    else:
      _, h_in, w_in, _ = input_shape

    if self.padding == "SAME":
      h_out = math.ceil(h_in / self.strides[0])
      w_out = math.ceil(w_in / self.strides[1])
    else:
      h_out = math.ceil((h_in - self.pool_size[0] + 1) / self.strides[0])
      w_out = math.ceil((w_in - self.pool_size[1] + 1) / self.strides[1])
    return [input_shape[0], input_shape[1], h_out, w_out]


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

  def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(MaxPooling2D, self).__init__(
        self.prot.maxpool2d,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, **kwargs)


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

  def __init__(self,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format=None,
               **kwargs):
    super(AveragePooling2D, self).__init__(
        self.prot.avgpool2d,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, **kwargs)
