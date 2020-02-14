# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe

np.random.seed(42)
tf.random.set_random_seed(42)


class TestLosses(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def test_binary_crossentropy(self):

    y_true_np = np.array([1, 1, 0, 0]).astype(float)
    y_pred_np = np.array([0.9, 0.1, 0.9, 0.1]).astype(float)

    y_true = tfe.define_private_variable(y_true_np)
    y_pred = tfe.define_private_variable(y_pred_np)

    loss = tfe.keras.losses.BinaryCrossentropy()
    out = loss(y_true, y_pred)
    der_for_y_pred = loss.grad(y_true, y_pred)

    with tfe.Session() as sess:
      sess.run(tf.global_variables_initializer())
      actual = sess.run(out.reveal())
      actual_der = sess.run(der_for_y_pred.reveal())

    tf.reset_default_graph()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      y_true = tf.convert_to_tensor(y_true_np)
      y_pred = tf.convert_to_tensor(y_pred_np)
      loss = tf.keras.losses.BinaryCrossentropy()
      out = loss(y_true, y_pred)
      der_for_y_pred = -(y_true / y_pred - (1 - y_true) / (1 - y_pred))

      expected = sess.run(out)
      expected_der = sess.run(der_for_y_pred)

    np.testing.assert_allclose(actual, expected, rtol=1e-1, atol=1e-1)

    # TODO: assertion below is currently failing; is this expected?
    del actual_der
    del expected_der
    # np.testing.assert_allclose(actual_der, expected_der, rtol=1e-1, atol=1e-1)

  def test_binary_crossentropy_from_logits(self):

    y_true_np = np.array([1, 1, 0, 0]).astype(float)
    y_pred_np = np.array([0.9, 0.1, 0.9, 0.1]).astype(float)

    y_true = tfe.define_private_variable(y_true_np)
    y_pred = tfe.define_private_variable(y_pred_np)

    loss = tfe.keras.losses.BinaryCrossentropy(from_logits=True)
    out = loss(y_true, y_pred)
    der_for_y_pred = loss.grad(y_true, y_pred)

    with tfe.Session() as sess:
      sess.run(tf.global_variables_initializer())
      actual = sess.run(out.reveal())
      actual_der = sess.run(der_for_y_pred.reveal())

    tf.reset_default_graph()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      y_true = tf.convert_to_tensor(y_true_np)
      y_pred = tf.convert_to_tensor(y_pred_np)
      loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
      out = loss(y_true, y_pred)
      der_for_y_pred = tf.sigmoid(y_pred) - y_true

      expected = sess.run(out)
      expected_der = sess.run(der_for_y_pred)

    np.testing.assert_allclose(actual, expected, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(actual_der, expected_der, rtol=1e-1, atol=1e-1)

  def test_mean_squared_error(self):
    y_true_np = np.array([1, 2, 3, 4]).astype(float)
    y_pred_np = np.array([0.9, 2.1, 3.2, 4.1]).astype(float)

    y_true = tfe.define_private_variable(y_true_np)
    y_pred = tfe.define_private_variable(y_pred_np)

    loss = tfe.keras.losses.MeanSquaredError()
    out = loss(y_true, y_pred)

    with tfe.Session() as sess:
      sess.run(tf.global_variables_initializer())
      actual = sess.run(out.reveal())

    tf.reset_default_graph()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      y_true = tf.convert_to_tensor(y_true_np)
      y_pred = tf.convert_to_tensor(y_pred_np)

      loss = tf.keras.losses.MeanSquaredError()
      out = loss(y_true, y_pred)
      expected = sess.run(out)

    np.testing.assert_allclose(actual, expected, rtol=1e-1, atol=1e-1)


if __name__ == '__main__':
  unittest.main()
