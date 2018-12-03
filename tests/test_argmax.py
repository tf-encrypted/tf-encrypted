import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestArgMax(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    @unittest.skipUnless(tfe.config.tensorflow_supports_int64(), "Too slow on Circle CI otherwise")
    def test_argmax_1d(self):

        t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(float)

        with tf.Session() as sess:
            out_tf = tf.argmax(t)
            expected = sess.run(out_tf)

        with tfe.protocol.SecureNN() as prot:
            out_tfe = prot.argmax(prot.define_private_variable(tf.constant(t)))

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                actual = sess.run(out_tfe.reveal())

        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(tfe.config.tensorflow_supports_int64(), "Too slow on Circle CI otherwise")
    def test_argmax_2d_axis0(self):

        t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 4).astype(float)

        with tf.Session() as sess:
            out_tf = tf.argmax(t, axis=0)
            expected = sess.run(out_tf)

        with tfe.protocol.SecureNN() as prot:
            out_tfe = prot.argmax(prot.define_private_variable(tf.constant(t)), axis=0)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                actual = sess.run(out_tfe.reveal())

        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(tfe.config.tensorflow_supports_int64(), "Too slow on Circle CI otherwise")
    def test_argmax_2d_axis1(self):

        t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 4).astype(float)

        with tf.Session() as sess:
            out_tf = tf.argmax(t, axis=1)
            expected = sess.run(out_tf)

        with tfe.protocol.SecureNN() as prot:
            out_tfe = prot.argmax(prot.define_private_variable(tf.constant(t)), axis=1)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                actual = sess.run(out_tfe.reveal())

        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(tfe.config.tensorflow_supports_int64(), "Too slow on Circle CI otherwise")
    def test_argmax_3d_axis0(self):

        t = np.array(np.arange(128)).reshape(8, 2, 2, 2, 2)

        with tf.Session() as sess:
            out = tf.argmax(t, axis=0)
            expected = sess.run(out)

        with tfe.protocol.SecureNN() as prot:
            out_tfe = prot.argmax(prot.define_private_variable(tf.constant(t)), axis=0)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                actual = sess.run(out_tfe.reveal())

        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
