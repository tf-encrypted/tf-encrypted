# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.testing_utils import agreement_test, layer_test

class TestActivation(unittest.TestCase):

  def setUp(self):
    tf.compat.v1.reset_default_graph()

  def test_activation_relu(self):
    self._core_activation(activation="relu")

  def _core_activation(self, **layer_kwargs):
    agreement_test(tfe.keras.layers.Activation,
                   kwargs=layer_kwargs,
                   input_shape=[1, 5])
    layer_test(tfe.keras.layers.Activation,
               kwargs=layer_kwargs,
               batch_input_shape=[1, 5])

  def test_backward_sigmoid(self) -> None:

    input_shape = [1, 4]
    input_data = np.array([-1, -0.75, 0.75, 1]).reshape(input_shape)
    weights_second_layer = np.ones(shape=[1])

    with tfe.protocol.SecureNN() as prot:

      private_input = prot.Variable(input_data)
      w = prot.Variable(weights_second_layer)

      tfe_layer = tfe.keras.layers.Activation("sigmoid", input_shape=[4])

      dense_out_pond = tfe_layer(private_input)

      loss = dense_out_pond * w

      # backward
      d_out = w
      _, d_x = tfe_layer.backward(d_out)

      with tfe.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tfe_loss = sess.run(loss.reveal())
        tfe_d_x = sess.run(d_x.reveal())

    # reset graph
    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as sess:

      tf_layer = tf.keras.layers.Activation("sigmoid", input_shape=[4])

      x = tf.Variable(input_data, dtype=tf.float32)
      y = tf_layer(x)
      w = tf.Variable(weights_second_layer, dtype=tf.float32)
      loss = y * w

      # backward
      d_x = tf.gradients(xs=x, ys=loss)

      sess.run(tf.compat.v1.global_variables_initializer())
      tf_loss, tf_d_x = sess.run([loss, d_x])

      np.testing.assert_array_almost_equal(tfe_loss, tf_loss, decimal=1)
      np.testing.assert_array_almost_equal(tfe_d_x, tf_d_x[0], decimal=2)


if __name__ == '__main__':
  unittest.main()
