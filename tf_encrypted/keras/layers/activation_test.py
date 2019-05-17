# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestActivation(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def test_activation_relu(self):
    self._core_activation(activation="relu")

  def _core_activation(self, **layer_kwargs):

    with tfe.protocol.SecureNN() as prot:

      input_shape = [1, 5]
      x = np.random.normal(size=input_shape)

      x_in = prot.define_private_variable(x)

      activation = tfe.keras.layers.Activation(**layer_kwargs)

      out = activation(x_in)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())

        out_pond = sess.run(out.reveal())

    #reset graph
    tf.reset_default_graph()

    with tf.Session() as sess:
      x_in = tf.Variable(x, dtype=tf.float32)

      activation_tf = tf.keras.layers.Activation(**layer_kwargs)

      out = activation_tf(x_in)

      sess.run(tf.global_variables_initializer())
      out_tensorflow = sess.run(out)

    np.testing.assert_array_almost_equal(out_pond, out_tensorflow, decimal=2)


if __name__ == '__main__':
  unittest.main()
