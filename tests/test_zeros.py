import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestZeros(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_zeros(self):

        with tfe.protocol.Pond() as prot:

            input_shape = [2, 3]

            tf.reset_default_graph()

            out = prot.zeros(input_shape)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                out_tfe = sess.run(out.reveal())

            tf.reset_default_graph()

            zeros_out_tf = tf.zeros(input_shape)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                out_tensorflow = sess.run(zeros_out_tf)

            np.testing.assert_allclose(out_tfe, out_tensorflow, atol=.01)


if __name__ == '__main__':
    unittest.main()
