import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

from tf_encrypted.layers.activation import Relu


class TestRelu(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_forward(self):
        input_shape = [4]
        input_relu = np.array([-1.0, -0.5, 0.5, 3.0]).astype(np.float32)

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

        assert(np.isclose(out_pond, out_tensorflow, atol=0.6).all())


if __name__ == '__main__':
    unittest.main()
