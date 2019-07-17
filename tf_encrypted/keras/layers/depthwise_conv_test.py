# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.testing_utils import agreement_test, layer_test

np.random.seed(42)


class TestDepthwiseConv2d(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def test_depthwise_conv2d_bias(self):
    self._core_depthwise_conv2d(kernel_size=2, use_bias=True)

  def test_depthwise_conv2d_nobias(self):
    self._core_depthwise_conv2d(kernel_size=2, use_bias=False)

  def test_depthwise_conv2d_same_padding(self):
    self._core_depthwise_conv2d(kernel_size=2, padding='same')

  def test_depthwise_conv2d_kernelsize_tuple(self):
    self._core_depthwise_conv2d(kernel_size=(2, 2))

  def test_depthwise_conv2d_depth_multiplier(self):
    self._core_depthwise_conv2d(kernel_size=2, depth_multiplier=2)

  def _core_depthwise_conv2d(self, **layer_kwargs):
    filters_in = 3
    input_shape = [2, 6, 6, filters_in]  # channels last

    if isinstance(layer_kwargs['kernel_size'], int):
      kernel_size_in = (layer_kwargs['kernel_size'],) * 2
    else:
      kernel_size_in = layer_kwargs['kernel_size']

    filters_out = layer_kwargs.get('depth_multiplier', 1)

    kernel = np.random.normal(kernel_size_in +
                              (filters_in, filters_out))

    initializer = tf.keras.initializers.Constant(kernel)

    base_kwargs = {
        "strides": 2,
        "depthwise_initializer": initializer,
    }

    kwargs = {**base_kwargs, **layer_kwargs}
    agreement_test(tfe.keras.layers.DepthwiseConv2D,
                   kwargs=kwargs,
                   input_shape=input_shape,
                   atol=1e-3)
    layer_test(tfe.keras.layers.DepthwiseConv2D,
               kwargs=kwargs,
               batch_input_shape=input_shape)


if __name__ == '__main__':
  unittest.main()
