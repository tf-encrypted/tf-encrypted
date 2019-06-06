# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.layers.activation import Relu, Sigmoid, Tanh


class TestRelu(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_forward(self):
    input_shape = [2, 2, 2, 50]
    input_relu = np.random.randn(np.prod(input_shape)).astype(
        np.float32).reshape(input_shape)

    with tfe.protocol.SecureNN() as prot:

      tf.reset_default_graph()

      relu_input = prot.define_private_variable(input_relu)
      relu_layer = Relu(input_shape)
      relu_out_pond = relu_layer.forward(relu_input)
      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out_pond = sess.run(relu_out_pond.reveal(), tag='tfe')

      tf.reset_default_graph()

      x = tf.Variable(input_relu, dtype=tf.float32)
      relu_out_tf = tf.nn.relu(x)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out_tensorflow = sess.run(relu_out_tf)

      np.testing.assert_allclose(out_pond, out_tensorflow, atol=.01)


class TestSigmoid(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_forward(self):
    input_shape = [4]
    input_sigmoid = np.array([-1.0, -0.5, 0.5, 3.0]).astype(np.float32)

    # sigmoid pond
    with tfe.protocol.Pond() as prot:

      sigmoid_input = prot.define_private_variable(input_sigmoid)
      sigmoid_layer = Sigmoid(input_shape)

      sigmoid_out_pond = sigmoid_layer.forward(sigmoid_input)

      with tfe.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # outputs
        out_pond = sess.run(sigmoid_out_pond.reveal())

      # reset graph
      tf.reset_default_graph()

      with tf.Session() as sess:
        x = tf.Variable(input_sigmoid, dtype=tf.float32)

        sigmoid_out_tf = tf.nn.sigmoid(x)

        sess.run(tf.global_variables_initializer())

        out_tensorflow = sess.run(sigmoid_out_tf)

    assert np.isclose(out_pond, out_tensorflow, atol=0.6).all()


class TestTanh(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_forward(self):
    input_shape = [4]
    input_tanh = np.array([-1.0, -0.5, 0.5, 3.0]).astype(np.float32)

    # tanh pond
    with tfe.protocol.Pond() as prot:

      tanh_input = prot.define_private_variable(input_tanh)
      tanh_layer = Tanh(input_shape)

      tanh_out_pond = tanh_layer.forward(tanh_input)

      with tfe.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # outputs
        out_pond = sess.run(tanh_out_pond.reveal())

      # reset graph
      tf.reset_default_graph()

      with tf.Session() as sess:
        x = tf.Variable(input_tanh, dtype=tf.float32)

        tanh_out_tf = tf.nn.tanh(x)

        sess.run(tf.global_variables_initializer())

        out_tensorflow = sess.run(tanh_out_tf)

    assert np.isclose(out_pond, out_tensorflow, atol=0.2).all()


if __name__ == '__main__':
  unittest.main()
