# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.testing_utils import agreement_test

np.random.seed(42)


class TestConv2d(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def test_conv2d_bias(self):
    self._core_conv2d(use_bias=True)

  def test_conv2d_nobias(self):
    self._core_conv2d(use_bias=False)

  def _core_conv2d(self, **layer_kwargs):
    filters_in = 3
    input_shape = [2, filters_in, 6, 6]  # channels first
    filters = 5
    kernel_size = 2
    padding = 'valid'
    kernel = np.random.normal((kernel_size, kernel_size) +
                              (filters_in, filters))
    initializer = tf.keras.initializers.Constant(kernel)

    base_kwargs = {
        "filters": filters,
        "kernel_size": kernel_size,
        "strides": 2,
        "kernel_initializer": initializer,
        "padding": padding,
    }

    kwargs = {**base_kwargs, **layer_kwargs}
    agreement_test(tfe.keras.layers.Conv2D,
                   kwargs=kwargs,
                   input_shape=input_shape)


if __name__ == '__main__':
  unittest.main()
