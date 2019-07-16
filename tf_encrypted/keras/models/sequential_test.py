# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras import backend as KE
from tf_encrypted.keras import Sequential
from tf_encrypted.keras.layers import Dense

np.random.seed(42)
tf.random.set_random_seed(42)


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
      x = tfe.define_private_input(
          "inputter",
          lambda: tf.convert_to_tensor(input_data))

      tfe_model = tfe.keras.models.model_from_config(k_config)
      tfe_model.set_weights(k_weights)
      y = tfe_model(x)

    with KE.get_session() as sess:
      actual = sess.run(y.reveal())

      np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)

    KE.clear_session()

  def test_from_config(self):
    input_shape = (1, 3)
    input_data = np.random.normal(size=input_shape)
    expected, k_weights, k_config = _model_predict_keras(input_data,
                                                         input_shape)

    with tfe.protocol.SecureNN():
      x = tfe.define_private_input(
          "inputter",
          lambda: tf.convert_to_tensor(input_data))

      tfe_model = Sequential.from_config(k_config)
      tfe_model.set_weights(k_weights)
      y = tfe_model(x)

    with KE.get_session() as sess:
      actual = sess.run(y.reveal())

      np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)

    KE.clear_session()

  def test_clone_model(self):
    input_shape = (1, 3)
    input_data = np.random.normal(size=input_shape)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2, batch_input_shape=input_shape))
    model.add(tf.keras.layers.Dense(3))
    expected = model.predict(input_data)

    with tfe.protocol.SecureNN():
      x = tfe.define_private_input(
          "inputter",
          lambda: tf.convert_to_tensor(input_data))

      tfe_model = tfe.keras.models.clone_model(model)

    with KE.get_session() as sess:
      y = tfe_model(x)
      actual = sess.run(y.reveal())

      np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)

    KE.clear_session()

  def test_weights_as_private_var(self):
    input_shape = (1, 3)
    input_data = np.random.normal(size=input_shape)
    expected, k_weights, k_config = _model_predict_keras(input_data,
                                                         input_shape)

    with tfe.protocol.SecureNN():
      x = tfe.define_private_input(
          "inputter",
          lambda: tf.convert_to_tensor(input_data))

      tfe_model = tfe.keras.models.model_from_config(k_config)
      weights_private_var = [tfe.define_private_variable(w) for w in k_weights]

      with tfe.Session() as sess:
        for w in weights_private_var:
          sess.run(w.initializer)

        tfe_model.set_weights(weights_private_var, sess)
        y = tfe_model(x)

        actual = sess.run(y.reveal())

        np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)


  def test_conv_model(self):

    num_classes = 10
    input_shape = (1, 28, 28, 1)
    input_data = np.random.normal(size=input_shape)

    with tf.Session():
      model = tf.keras.models.Sequential()

      model.add(tf.keras.layers.Conv2D(2,
                                       (3, 3),
                                       batch_input_shape=input_shape))
      model.add(tf.keras.layers.AveragePooling2D((2, 2)))
      model.add(tf.keras.layers.Conv2D(2, (3, 3)))
      model.add(tf.keras.layers.AveragePooling2D((2, 2)))
      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.Dense(num_classes, name="logit"))

      expected = model.predict(input_data)
      k_weights = model.get_weights()
      k_config = model.get_config()

    with tfe.protocol.SecureNN():
      x = tfe.define_private_input(
          "inputter",
          lambda: tf.convert_to_tensor(input_data))

      tfe_model = tfe.keras.models.model_from_config(k_config)
      tfe_model.set_weights(k_weights)
      y = tfe_model(x)

    with KE.get_session() as sess:
      actual = sess.run(y.reveal())

      np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)

    KE.clear_session()

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
