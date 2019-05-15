# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestDense(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def test_dense_bias(self):
    self._core_dense(use_bias=True)

  def test_dense_nobias(self):
    self._core_dense(use_bias=False)

  def _core_dense(self, **layer_kwargs):

    with tfe.protocol.Pond() as prot:

      input_shape = [4, 5]
      x = np.random.normal(size=input_shape)

      kernel_shape = [5, 4]
      kernel_values = np.random.normal(size=kernel_shape)
      initializer = tf.keras.initializers.Constant(value=kernel_values)

      x_in = prot.define_private_variable(x)

      fc = tfe.keras.layers.Dense(
          4, kernel_initializer=initializer, **layer_kwargs,
      )

      out = fc(x_in)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())

        out_pond = sess.run(out.reveal())

    #reset graph
    tf.reset_default_graph()

    with tf.Session() as sess:
      x_in = tf.Variable(x, dtype=tf.float32)

      fc_tf = tf.keras.layers.Dense(
          4, kernel_initializer=initializer, **layer_kwargs,
      )

      out = fc_tf(x_in)

      sess.run(tf.global_variables_initializer())
      out_tensorflow = sess.run(out)

    np.testing.assert_array_almost_equal(out_pond, out_tensorflow, decimal=2)


if __name__ == '__main__':
  unittest.main()
