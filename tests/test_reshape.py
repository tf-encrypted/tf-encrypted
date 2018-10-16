import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.layers import Reshape


class TestReshape(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_forward(self):
        input_shape = [2, 3, 4, 5]
        output_shape = [2, -1]
        input_reshape = np.random.standard_normal(input_shape)

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        # reshape pond
        with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:

            reshape_input = prot.define_private_variable(input_reshape)
            reshape_layer = Reshape(input_shape, output_shape)

            reshape_out_pond = reshape_layer.forward(reshape_input)

            with tfe.Session() as sess:

                sess.run(tf.global_variables_initializer())
                # outputs
                out_pond = sess.run(reshape_out_pond.reveal())

            # reset graph
            tf.reset_default_graph()

            with tf.Session() as sess:
                x = tf.Variable(input_reshape, dtype=tf.float32)

                reshape_out_tf = tf.reshape(x, output_shape)

                sess.run(tf.global_variables_initializer())

                out_tensorflow = sess.run(reshape_out_tf)

        assert(np.isclose(out_pond, out_tensorflow, atol=0.6).all())


if __name__ == '__main__':
    unittest.main()
