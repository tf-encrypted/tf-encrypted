import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestMatMul(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_matmul(self) -> None:

        with tfe.protocol.Pond() as prot:

            input_shape = [4, 5]
            input = np.random.normal(size=input_shape)

            filter_shape = [5, 4]
            filter_values = np.random.normal(size=filter_shape)

            input_input = prot.define_private_variable(input)
            filter_filter = prot.define_private_variable(filter_values)

            out = prot.matmul(input_input, filter_filter)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())

                out_pond = sess.run(out.reveal())

        # reset graph
        tf.reset_default_graph()

        with tf.Session() as sess:
            x = tf.Variable(input, dtype=tf.float32)
            filters_tf = tf.Variable(filter_values, dtype=tf.float32)

            out = tf.matmul(x, filters_tf)

            sess.run(tf.global_variables_initializer())
            out_tensorflow = sess.run(out)

        np.testing.assert_array_almost_equal(out_pond, out_tensorflow, decimal=2)

    def test_big_middle_matmul(self) -> None:
        with tfe.protocol.Pond() as prot:

            input_shape = [64, 4500]
            input = np.random.normal(size=input_shape)

            filter_shape = [4500, 64]
            filter_values = np.random.normal(size=filter_shape)

            input_input = prot.define_private_variable(input)
            filter_filter = prot.define_private_variable(filter_values)

            out = prot.matmul(input_input, filter_filter)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())

                out_pond = sess.run(out.reveal())

        # reset graph
        tf.reset_default_graph()

        with tf.Session() as sess:
            x = tf.Variable(input, dtype=tf.float32)
            filters_tf = tf.Variable(filter_values, dtype=tf.float32)

            out = tf.matmul(x, filters_tf)

            sess.run(tf.global_variables_initializer())
            out_tensorflow = sess.run(out)

        np.testing.assert_allclose(out_pond, out_tensorflow, atol=.1)


if __name__ == '__main__':
    unittest.main()
