import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.layers.activation import Relu


class TestRelu(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_forward(self):
        input_shape = [2, 2, 2, 50]
        input_relu = np.random.randn(np.prod(input_shape)).astype(np.float32).reshape(input_shape)

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

        np.testing.assert_allclose(out_pond, out_tensorflow, atol=.01), out_pond


if __name__ == '__main__':
    unittest.main()
