# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras import Sequential
from tf_encrypted.keras.layers import Dense

np.random.seed(42)


class TestSequential(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_two_layers(self):
    shape = (1, 3)
    input_data = np.random.normal(size=shape)
    with tfe.protocol.SecureNN():
      model = Sequential()
      model.add(Dense(2, input_shape=shape))
      model.add(Dense(3))

      x = tfe.define_private_variable(input_data)
      model(x)


if __name__ == '__main__':
  unittest.main()
