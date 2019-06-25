# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

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

  def test_model_from_config(self):
    input_shape = (1, 3)
    input_data = np.random.normal(size=input_shape)
    expected, k_weights, k_config = _model_predict_keras(input_data,
                                                         input_shape)

    with tfe.protocol.SecureNN():
      tfe_model = tfe.keras.models.model_from_config(k_config)
      x = tfe.define_private_variable(input_data)

    with tfe.Session() as sess:
      sess.run(tf.global_variables_initializer())
      tfe_model.set_weights(k_weights)
      y = tfe_model(x)
      actual = sess.run(y.reveal())

      np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)

  def test_from_config(self):
    input_shape = (1, 3)
    input_data = np.random.normal(size=input_shape)
    expected, k_weights, k_config = _model_predict_keras(input_data,
                                                         input_shape)

    with tfe.protocol.SecureNN():
      tfe_model = tfe.keras.models.Sequential([])
      tfe_model = tfe_model.from_config(k_config)
      x = tfe.define_private_variable(input_data)

    with tfe.Session() as sess:
      sess.run(tf.global_variables_initializer())
      tfe_model.set_weights(k_weights)
      y = tfe_model(x)
      actual = sess.run(y.reveal())

      np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)

  def test_clone_model(self):
    input_shape = (1, 3)
    input_data = np.random.normal(size=input_shape)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2, batch_input_shape=input_shape))
    model.add(tf.keras.layers.Dense(3))
    expected = model.predict(input_data)

    with tfe.protocol.SecureNN():
      tfe_model = tfe.keras.models.clone_model(model)
      x = tfe.define_private_variable(input_data)

    with K.get_session() as sess:
      # won't work if we re-initialize all the weights
      # with sess.run(tf.global_variables_initializer())
      sess.run(x.initializer)
      y = tfe_model(x)
      actual = sess.run(y.reveal())

      np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)

  def test_weights_as_private_var(self):
    input_shape = (1, 3)
    input_data = np.random.normal(size=input_shape)
    expected, k_weights, k_config = _model_predict_keras(input_data,
                                                         input_shape)

    with tfe.protocol.SecureNN():
      tfe_model = tfe.keras.models.model_from_config(k_config)
      weights_private_var = [tfe.define_private_variable(w) for w in k_weights]
      print(type(weights_private_var[0]))
      x = tfe.define_private_variable(input_data)

    with tfe.Session() as sess:
      sess.run(tf.global_variables_initializer())
      tfe_model.set_weights(weights_private_var)
      y = tfe_model(x)
      actual = sess.run(y.reveal())

      np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-4)


def _model_predict_keras(input_data, input_shape):
  with tf.Session():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2, batch_input_shape=input_shape))
    model.add(tf.keras.layers.Dense(3))

    weights = model.get_weights()
    config = model.get_config()
    out = model.predict(input_data)

  return out, weights, config


if __name__ == '__main__':
  unittest.main()
