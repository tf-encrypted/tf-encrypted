import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe


class TestMatMul(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_matmul(self) -> None:
        config = tfe.LocalConfig([
            'server_0',
            'server_1',
            'crypto_producer'
        ])

        with tfe.protocol.Pond(*config.get_players('server_0, server_1, crypto_producer')) as prot:
            input_shape = [4, 5]
            input = np.random.normal(size=input_shape)

            filter_shape = [5, 4]
            filter_values = np.random.normal(size=filter_shape)

            input_input = prot.define_private_variable(input)
            filter_filter = prot.define_private_variable(filter_values)

            out = prot.dot(input_input, filter_filter)
            out2 = prot.matmul(input_input, filter_filter)

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())

                out_pond = out.reveal().eval(sess)
                out_pond2 = out2.reveal().eval(sess)

        # reset graph
        tf.reset_default_graph()

        with tf.Session() as sess:
            x = tf.Variable(input, dtype=tf.float32)
            filters_tf = tf.Variable(filter_values, dtype=tf.float32)

            out = tf.matmul(x, filters_tf)

            sess.run(tf.global_variables_initializer())
            out_tensorflow = sess.run(out)

        np.testing.assert_array_almost_equal(out_pond, out_tensorflow, decimal=2)
        np.testing.assert_array_almost_equal(out_pond2, out_tensorflow, decimal=2)

    def test_big_middle_matmul(self) -> None:
        config = tfe.LocalConfig([
            'server_0',
            'server_1',
            'crypto_producer'
        ])

        with tfe.protocol.Pond(*config.get_players('server_0, server_1, crypto_producer')) as prot:
            input_shape = [64, 4500]
            input = np.random.normal(size=input_shape)

            filter_shape = [4500, 64]
            filter_values = np.random.normal(size=filter_shape)

            input_input = prot.define_private_variable(input)
            filter_filter = prot.define_private_variable(filter_values)

            out = prot.dot(input_input, filter_filter)

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())

                out_pond = out.reveal().eval(sess)

        # reset graph
        tf.reset_default_graph()

        with tf.Session() as sess:
            x = tf.Variable(input, dtype=tf.float32)
            filters_tf = tf.Variable(filter_values, dtype=tf.float32)

            out = tf.matmul(x, filters_tf)

            sess.run(tf.global_variables_initializer())
            out_tensorflow = sess.run(out)

        np.testing.assert_array_almost_equal(out_pond, out_tensorflow, decimal=2)


if __name__ == '__main__':
    unittest.main()
