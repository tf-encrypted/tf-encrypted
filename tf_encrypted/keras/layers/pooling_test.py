# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.testing_utils import agreement_test

np.random.seed(42)


class TestPooling2d(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def test_maxpooling2d_valid(self):
    self._core_maxpooling2d(padding='valid')

  def test_avgpooling2d_valid(self):
    self._core_avgpooling2d(padding='valid')

  def _core_maxpooling2d(self, **layer_kwargs):
    channel_in = 2
    input_shape = [2, 8, 8, channel_in]  # channels last
    pool_size_in = 2
    strides_in = 2

    base_kwargs = {
        "pool_size": pool_size_in,
        "strides": strides_in,
    }

    kwargs = {**base_kwargs, **layer_kwargs}
    agreement_test(tfe.keras.layers.MaxPooling2D,
                   kwargs=kwargs,
                   input_shape=input_shape)

  def _core_avgpooling2d(self, **layer_kwargs):
    channel_in = 2
    input_shape = [2, 8, 8, channel_in]  # channels last
    pool_size_in = 2
    strides_in = 2

    base_kwargs = {
        "pool_size": pool_size_in,
        "strides": strides_in,
    }

    kwargs = {**base_kwargs, **layer_kwargs}
    agreement_test(tfe.keras.layers.AveragePooling2D,
                   kwargs=kwargs,
                   input_shape=input_shape)


if __name__ == '__main__':
  unittest.main()
