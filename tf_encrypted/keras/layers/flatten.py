"""Flatten Layer implementation."""

import numpy as np
from tensorflow.python.keras.utils import conv_utils

import tf_encrypted as tfe
from tf_encrypted.keras.engine import Layer


class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.
  If inputs are shaped `(batch,)` without a channel dimension, then flattening
  adds an extra channel dimension and output shapes are `(batch, 1)`.
  Arguments:
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, ..., channels)` while `channels_first` corresponds to
          inputs with shape `(batch, channels, ...)`.
          If you never set it, then it will be "channels_last".
  """

    def __init__(self, data_format=None, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        self._layer_input = inputs
        input_shape = inputs.shape.as_list()
        rank = len(input_shape)

        if self.data_format == "channels_first" and rank > 1:
            permutation = [0]
            permutation.extend(i for i in range(2, rank))
            permutation.append(1)
            inputs = tfe.transpose(inputs, perm=permutation)

        if rank == 1:
            flatten_shape = [input_shape[0], 1]
        else:
            flatten_shape = [input_shape[0], -1]

        outputs = tfe.reshape(inputs, flatten_shape)
        self._layer_output = outputs
        return outputs

    def backward(self, d_y):
        x = self._layer_input
        grad_weights = []

        input_shape = x.shape.as_list()

        d_x = tfe.reshape(d_y, input_shape)
        return grad_weights, d_x

    def compute_output_shape(self, input_shape):
        if not input_shape:
            raise ValueError("input_shape shouldn't be empty or None")
        output_shape = [input_shape[0]]
        if all(input_shape[1:]):
            output_shape += [np.prod(input_shape[1:])]
        else:
            output_shape += [None]
        return output_shape
