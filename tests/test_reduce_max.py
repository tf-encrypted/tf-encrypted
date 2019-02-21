import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
import pytest


@pytest.mark.slow
class TestReduceMax(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def tearDown(self):
        tf.reset_default_graph()

    def test_reduce_max_1d(self):

        t = np.array([1, 2, 3, 4]).astype(float)

        with tf.Session() as sess:
            out_tf = tf.reduce_max(t)
            expected = sess.run(out_tf)

        with tfe.protocol.SecureNN() as prot:
            b = prot.define_private_variable(tf.constant(t))
            out_tfe = prot.reduce_max(b)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for _ in range(2):
                    actual = sess.run(out_tfe.reveal(), tag='test_1d')

        np.testing.assert_array_equal(actual, expected)

    def test_reduce_max_2d_axis0(self):

        t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 4).astype(float)

        with tf.Session() as sess:
            out_tf = tf.reduce_max(t, axis=0)
            expected = sess.run(out_tf)

        with tfe.protocol.SecureNN() as prot:
            b = prot.define_private_variable(tf.constant(t))
            out_tfe = prot.reduce_max(b, axis=0)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for _ in range(2):
                    actual = sess.run(out_tfe.reveal(), tag='test_2d_axis0')

        np.testing.assert_array_equal(actual, expected)

    def test_reduce_max_2d_axis1(self):

        t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 4).astype(float)

        with tf.Session() as sess:
            out_tf = tf.reduce_max(t, axis=1)
            expected = sess.run(out_tf)

        with tfe.protocol.SecureNN() as prot:
            b = prot.define_private_variable(tf.constant(t))
            out_tfe = prot.reduce_max(b, axis=1)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for _ in range(2):
                    actual = sess.run(out_tfe.reveal(), tag='test_2d_axis1')

        np.testing.assert_array_equal(actual, expected)

    def test_reduce_max_3d_axis0(self):

        t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 2, 2)

        with tf.Session() as sess:
            out = tf.reduce_max(t, axis=0)
            expected = sess.run(out)

        with tfe.protocol.SecureNN() as prot:
            b = prot.define_private_variable(tf.constant(t))
            out_tfe = prot.reduce_max(b, axis=0)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for _ in range(2):
                    actual = sess.run(out_tfe.reveal(), tag='test_3d_axis0')

        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
