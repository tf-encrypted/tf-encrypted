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

  def test_backward(self):
    input_shape = [1, 5]
    input_data = np.ones(input_shape)
    weights_second_layer = np.ones(shape=[1, 5])
    kernel = np.ones([5, 5])
    initializer = tf.keras.initializers.Constant(kernel)

    with tfe.protocol.SecureNN() as prot:

      private_input = prot.define_private_variable(input_data)
      w = prot.define_private_variable(weights_second_layer)

      tfe_layer = tfe.keras.layers.Dense(5,
                                         input_shape=input_shape[1:],
                                         kernel_initializer=initializer)

      dense_out_pond = tfe_layer(private_input)

      loss = dense_out_pond * w

      # backward
      d_out = w
      grad, d_x = tfe_layer.backward(d_out)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tfe_loss = sess.run(loss.reveal())
        tfe_d_k = sess.run(grad[0].reveal())
        tfe_d_b = sess.run(grad[1].reveal())
        tfe_d_x = sess.run(d_x.reveal())

    # reset graph
    tf.reset_default_graph()

    with tf.Session() as sess:

      initializer = tf.keras.initializers.Constant(kernel)

      tf_layer = tf.keras.layers.Dense(5,
                                       input_shape=input_shape[1:],
                                       kernel_initializer=initializer)
      x = tf.Variable(input_data, dtype=tf.float32)
      y = tf_layer(x)

      w = tf.Variable(weights_second_layer, dtype=tf.float32)
      loss = y * w
      k, b = tf_layer.trainable_weights

      # backward
      d_x, d_k, d_b = tf.gradients(xs=[x, k, b], ys=loss)

      sess.run(tf.global_variables_initializer())
      tf_loss, tf_d_x, tf_d_k, tf_d_b = sess.run([loss, d_x, d_k, d_b])

      np.testing.assert_array_almost_equal(tfe_loss, tf_loss, decimal=2)
      np.testing.assert_array_almost_equal(tfe_d_k, tf_d_k, decimal=2)
      np.testing.assert_array_almost_equal(tfe_d_b, tf_d_b, decimal=2)
      np.testing.assert_array_almost_equal(tfe_d_x, tf_d_x, decimal=2)


if __name__ == '__main__':
  unittest.main()
