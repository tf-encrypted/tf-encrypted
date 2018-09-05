import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.layers.activation import Sigmoid


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_forward(self):
        input_shape = [4]
        input_sigmoid = np.array([-1.0, -0.5, 0.5, 3.0]).astype(np.float32)

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        # sigmoid pond
        with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:

            sigmoid_input = prot.define_private_variable(input_sigmoid)
            sigmoid_layer = Sigmoid(input_shape)

            sigmoid_out_pond = sigmoid_layer.forward(sigmoid_input)

            with config.session() as sess:

                sess.run(tf.global_variables_initializer())
                # outputs
                out_pond = sigmoid_out_pond.reveal().eval(sess)

            # reset graph
            tf.reset_default_graph()

            with tf.Session() as sess:
                x = tf.Variable(input_sigmoid, dtype=tf.float32)

                sigmoid_out_tf = tf.nn.sigmoid(x)

                sess.run(tf.global_variables_initializer())

                out_tensorflow = sess.run(sigmoid_out_tf)

        assert(np.isclose(out_pond, out_tensorflow, atol=0.6).all())


if __name__ == '__main__':
    unittest.main()
