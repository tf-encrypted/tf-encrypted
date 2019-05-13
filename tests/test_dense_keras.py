# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestDense(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_dense(self) -> None:

    with tfe.protocol.Pond() as prot:

      input_shape = [4, 5]
      x_in = np.random.normal(size=input_shape)

      filter_shape = [5, 4]
      filter_values = np.random.normal(size=filter_shape)

      input_input = prot.define_private_variable(x_in)

      fc = tfe.keras.layers.Dense(4,
                                  use_bias=False,
                                  kernel_initializer=filter_values,
                                  input_shape=input_shape)

      out = fc(input_input)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())

        out_pond = sess.run(out.reveal())

    #reset graph
    tf.reset_default_graph()

    with tf.Session() as sess:
      x = tf.Variable(x_in, dtype=tf.float32)
      filter = tf.keras.initializers.Constant(value=filter_values)

      fc_tf = tf.keras.layers.Dense(4,
                                    use_bias=False,
                                    kernel_initializer=filter)

      out = fc_tf(x)

      sess.run(tf.global_variables_initializer())
      out_tensorflow = sess.run(out)

    np.testing.assert_array_almost_equal(out_pond, out_tensorflow, decimal=2)


if __name__ == '__main__':
  unittest.main()
