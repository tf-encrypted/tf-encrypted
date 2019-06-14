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

  def test_keras_to_tfe(self):
    shape = (1, 3)
    input_data = np.random.normal(size=shape)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2, batch_input_shape=shape))
    model.add(tf.keras.layers.Dense(3))

    k_weights = model.get_weights()
    k_config = model.get_config()

    expected = model.predict(input_data)

    with tfe.protocol.SecureNN():
      tfe_model = tfe.keras.Sequential.from_config(k_config)
      x = tfe.define_private_variable(input_data)

    with tfe.Session() as sess:
      sess.run(tf.global_variables_initializer())
      tfe_model.set_weights(k_weights, sess)
      y = tfe_model(x)
      actual = sess.run(y.reveal())

      np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-8)


if __name__ == '__main__':
  unittest.main()
