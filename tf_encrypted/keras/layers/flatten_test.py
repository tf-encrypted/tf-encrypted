# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.testing_utils import agreement_test

np.random.seed(42)


class TestFlatten(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def test_flatten_rank_four(self):
    self._core_flatten(input_shape=[4, 5, 2, 2])

  def test_flatten_rank_one(self):
    self._core_flatten(input_shape=[4])

  def test_flatten_channels_first(self):
    self._core_flatten(input_shape=[4, 5, 2, 2],
                       data_format='channels_first')

  def _core_flatten(self, **layer_kwargs):
    input_shape = layer_kwargs['input_shape']

    agreement_test(tfe.keras.layers.Flatten,
                   kwargs=layer_kwargs,
                   input_shape=input_shape,
                   atol=.1)


if __name__ == '__main__':
  unittest.main()
