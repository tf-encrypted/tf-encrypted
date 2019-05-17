# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestConv2d(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def test_conv2d_bias(self):
    self._core_conv2d(use_bias=True)

  def test_conv2d_nobias(self):
    self._core_conv2d(use_bias=False)

  def _core_conv2d(self, **layer_kwargs):

    with tfe.protocol.SecureNN() as prot:

      batch_size, channels_in, filters = 32, 3, 64
      img_height, img_width = 28, 28
      input_shape = (batch_size, channels_in, img_height, img_width)
      x = np.random.normal(size=input_shape).astype(np.float32)

      # kernel
      strides = 2
      kernel_size = (2, 2)
      kernel_shape = kernel_size + (channels_in, filters)
      kernel_values = np.random.normal(size=kernel_shape)

      initializer = tf.keras.initializers.Constant(value=kernel_values)

      x_in = prot.define_private_variable(x)

      conv = tfe.keras.layers.Conv2D(
          filters,
          kernel_size,
          strides,
          kernel_initializer=initializer, **layer_kwargs,
      )

      out = conv(x_in)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())

        out_pond = sess.run(out.reveal())

    #reset graph
    tf.reset_default_graph()

    with tf.Session() as sess:
      x_in = tf.Variable(x, dtype=tf.float32)

      conv_tf = tf.keras.layers.Conv2D(
          filters,
          kernel_size,
          strides,
          kernel_initializer=initializer, **layer_kwargs,
      )

      out = conv_tf(x_in)

      sess.run(tf.global_variables_initializer())
      out_tensorflow = sess.run(out)

    np.testing.assert_array_almost_equal(out_pond, out_tensorflow, decimal=2)


if __name__ == '__main__':
  unittest.main()
