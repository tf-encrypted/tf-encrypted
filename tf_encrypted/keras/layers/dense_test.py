# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.testing_utils import agreement_test, layer_test

np.random.seed(42)


class TestDense(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def test_dense_bias(self):
    self._core_dense(use_bias=True)

  def test_dense_nobias(self):
    self._core_dense(use_bias=False)

  def test_dense_relu(self):
    self._core_dense(activation="relu")

  def _core_dense(self, **layer_kwargs):
    input_shape = [4, 5]
    kernel = np.random.normal(input_shape[::-1])
    initializer = tf.keras.initializers.Constant(kernel)

    base_kwargs = {
        "units": 4,
        "kernel_initializer": initializer,
    }
    kwargs = {**base_kwargs, **layer_kwargs}
    agreement_test(tfe.keras.layers.Dense,
                   kwargs=kwargs,
                   input_shape=input_shape)
    layer_test(tfe.keras.layers.Dense,
               kwargs=kwargs,
               batch_input_shape=input_shape)


if __name__ == '__main__':
  unittest.main()
