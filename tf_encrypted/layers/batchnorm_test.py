# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

from tf_encrypted.layers import Batchnorm


class TestBatchnorm(unittest.TestCase):
  def setUp(self):
    tf.compat.v1.reset_default_graph()

  def test_channels_first(self) -> None:
    """
    Test batch norm layer with NCHW (channels first) format
    """
    channels_first = True

    batch_size, channels_in, img_height, img_width = (32, 3, 28, 28)

    input_shape = [batch_size, channels_in, img_height, img_width]
    input_batchnorm = np.random.normal(size=input_shape).astype(np.float32)

    # I reshaped the input because tf.nn.batch_normalization doesn't reshape it
    # automatically However tf encrypted will reshape automatically the input
    shape = (1, channels_in, 1, 1)
    mean = np.array([2.0, 1.5, 20.8]).reshape(shape).astype(np.float32)
    variance = np.array([0.5, 0.3, 0.1]).reshape(shape).astype(np.float32)
    scale = np.array([0.3, 0.5, 0.8]).reshape(shape).astype(np.float32)
    offset = np.array([1.5, 1.2, 1.4]).reshape(shape).astype(np.float32)
    variance_epsilon = 1e-8

    with tfe.protocol.Pond() as prot:
      batchnorm_input = prot.Variable(input_batchnorm)

      batchnorm_layer = Batchnorm(
          input_shape,
          mean,
          variance,
          scale,
          offset,
          channels_first=channels_first)
      batchnorm_layer.initialize()
      batchnorm_out_pond = batchnorm_layer.forward(batchnorm_input)

      with tfe.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        out_pond = sess.run(batchnorm_out_pond.reveal())

      # reset graph
      tf.compat.v1.reset_default_graph()

      with tf.compat.v1.Session() as sess:
        x = tf.Variable(input_batchnorm, dtype=tf.float32)

        batchnorm_out_tf = tf.nn.batch_normalization(
            x, mean, variance, offset, scale, variance_epsilon)

        sess.run(tf.compat.v1.global_variables_initializer())

        out_tensorflow = sess.run(batchnorm_out_tf)

        np.testing.assert_array_almost_equal(
            out_pond, out_tensorflow, decimal=1)

  def test_channels_last(self) -> None:
    """
    Test batch norm layer with NHWC (channels last) format
    """
    channels_first = False

    batch_size, img_height, img_width, channels_in = (32, 28, 28, 3)

    input_shape = [batch_size, img_height, img_width, channels_in]
    input_batchnorm = np.random.normal(size=input_shape).astype(np.float32)

    # I reshaped the input because tf.nn.batch_normalization doesn't reshape it
    # automatically However tf encrypted will reshape automatically the input
    shape = (1, 1, 1, channels_in)
    mean = np.array([2.0, 1.5, 20.8]).reshape(shape).astype(np.float32)
    variance = np.array([0.5, 0.3, 0.1]).reshape(shape).astype(np.float32)
    scale = np.array([0.3, 0.5, 0.8]).reshape(shape).astype(np.float32)
    offset = np.array([1.5, 1.2, 1.4]).reshape(shape).astype(np.float32)
    variance_epsilon = 1e-8

    with tfe.protocol.Pond() as prot:
      batchnorm_input = prot.Variable(input_batchnorm)

      batchnorm_layer = Batchnorm(
          input_shape,
          mean,
          variance,
          scale,
          offset,
          channels_first=channels_first)
      batchnorm_layer.initialize()
      batchnorm_out_pond = batchnorm_layer.forward(batchnorm_input)

      with tfe.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        out_pond = sess.run(batchnorm_out_pond.reveal())

      # reset graph
      tf.compat.v1.reset_default_graph()

      with tf.compat.v1.Session() as sess:
        x = tf.Variable(input_batchnorm, dtype=tf.float32)

        batchnorm_out_tf = tf.nn.batch_normalization(
            x, mean, variance, offset, scale, variance_epsilon)

        sess.run(tf.compat.v1.global_variables_initializer())

        out_tensorflow = sess.run(batchnorm_out_tf)

        np.testing.assert_array_almost_equal(
            out_pond, out_tensorflow, decimal=1)



if __name__ == '__main__':
  unittest.main()
