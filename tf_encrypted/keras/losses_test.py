# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
np.random.seed(42)
tf.compat.v1.random.set_random_seed(42)

class TestLosses(unittest.TestCase):
  def setUp(self):
    tf.compat.v1.reset_default_graph()

  def test_binary_crossentropy(self):

    y_true_np = np.array([1, 1, 0, 0]).astype(float)
    y_pred_np = np.array([0.9, 0.1, 0.9, 0.1]).astype(float)

    y_true = tfe.Variable(y_true_np)
    y_pred = tfe.Variable(y_pred_np)

    loss = tfe.keras.losses.BinaryCrossentropy()
    out = loss(y_true, y_pred)

    with tfe.Session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      actual = sess.run(out.reveal())

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      y_true = tf.convert_to_tensor(value=y_true_np)
      y_pred = tf.convert_to_tensor(value=y_pred_np)

      loss = tf.keras.losses.BinaryCrossentropy()
      out = loss(y_true, y_pred)
      expected = sess.run(out)

    np.testing.assert_allclose(actual, expected, rtol=1e-1, atol=1e-1)

  def test_mean_squared_error(self):
    y_true_np = np.array([1, 2, 3, 4]).astype(float)
    y_pred_np = np.array([0.9, 2.1, 3.2, 4.1]).astype(float)

    y_true = tfe.Variable(y_true_np)
    y_pred = tfe.Variable(y_pred_np)

    loss = tfe.keras.losses.MeanSquaredError()
    out = loss(y_true, y_pred)

    with tfe.Session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      actual = sess.run(out.reveal())

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      y_true = tf.convert_to_tensor(y_true_np)
      y_pred = tf.convert_to_tensor(y_pred_np)

      loss = tf.keras.losses.MeanSquaredError()
      out = loss(y_true, y_pred)
      expected = sess.run(out)

    np.testing.assert_allclose(actual, expected, rtol=1e-1, atol=1e-1)


if __name__ == '__main__':
  unittest.main()
