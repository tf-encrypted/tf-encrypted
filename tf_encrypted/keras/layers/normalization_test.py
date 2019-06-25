# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.testing_utils import agreement_test, layer_test

np.random.seed(42)


class TestBatchNormalization(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def test_batchnorm_2d(self):
    self._core_batchnorm([1, 4], axis=1)

  def test_batchnorm_4d(self):
    self._core_batchnorm([1, 1, 1, 3])

  def test_batchnorm_channels_first(self):
    self._core_batchnorm([1, 3, 1, 1], axis=1)

  def test_batchnorm_no_scale(self):
    self._core_batchnorm([1, 1, 1, 3], scale=False)

  def test_batchnorm_no_center(self):
    self._core_batchnorm([1, 1, 1, 3], center=False)

  def test_batchnorm_non_default_mean_init(self):
    input_shape = [1, 1, 3]
    const = np.random.normal(input_shape)
    initializer = tf.keras.initializers.Constant(const)

    self._core_batchnorm([1] + input_shape, moving_mean_initializer=initializer)

  def test_batchnorm_non_default_variance_init(self):
    input_shape = [1, 1, 3]
    const = np.random.uniform(input_shape)
    initializer = tf.keras.initializers.Constant(const)

    self._core_batchnorm([1] + input_shape,
                         moving_variance_initializer=initializer)

  def _core_batchnorm(self, input_shape, **layer_kwargs):
    base_kwargs = {
        'fused': False
    }

    kwargs = {**base_kwargs, **layer_kwargs}

    agreement_test(tfe.keras.layers.BatchNormalization,
                   kwargs=kwargs,
                   input_shape=input_shape)
    layer_test(tfe.keras.layers.BatchNormalization,
               kwargs=kwargs,
               batch_input_shape=input_shape)


if __name__ == '__main__':
  unittest.main()
