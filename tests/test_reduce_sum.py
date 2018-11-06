import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestReduceSum(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_reduce_sum_1d(self):

        t = [1, 2]
        with tf.Session() as sess:
            out = tf.reduce_sum(t)
            actual = sess.run(out)

        with tfe.protocol.Pond() as prot:
            b = prot.define_private_variable(tf.constant(t))
            out = prot.reduce_sum(b)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out.reveal())

        np.testing.assert_array_equal(final, actual)

    def test_reduce_sum_2d(self):

        t = [[1, 2], [1, 3]]
        with tf.Session() as sess:
            out = tf.reduce_sum(t, axis=1)
            actual = sess.run(out)

        with tfe.protocol.Pond() as prot:
            b = prot.define_private_variable(tf.constant(t))
            out = prot.reduce_sum(b, axis=1)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out.reveal())

        np.testing.assert_array_equal(final, actual)

    def test_reduce_sum_huge_vector(self):

        t = [1] * 2**13
        with tf.Session() as sess:
            out = tf.reduce_sum(t)
            actual = sess.run(out)

        with tfe.protocol.Pond() as prot:
            b = prot.define_private_variable(tf.constant(t))
            out = prot.reduce_sum(b)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out.reveal())

        np.testing.assert_array_equal(final, actual)


if __name__ == '__main__':
    unittest.main()
