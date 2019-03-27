import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestNegative(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_negative(self):
        input_shape = [2, 2]
        input_neg = np.ones(input_shape)

        # reshape pond
        with tfe.protocol.Pond() as prot:

            neg_input = prot.define_private_variable(input_neg)

            neg_out_pond = prot.negative(neg_input)

            with tfe.Session() as sess:

                sess.run(tf.global_variables_initializer())
                # outputs
                out_pond = sess.run(neg_out_pond.reveal())

            # reset graph
            tf.reset_default_graph()

            with tf.Session() as sess:
                x = tf.Variable(input_neg, dtype=tf.float32)

                neg_out_tf = tf.negative(x)

                sess.run(tf.global_variables_initializer())

                out_tensorflow = sess.run(neg_out_tf)

        assert(np.isclose(out_pond, out_tensorflow, atol=0.6).all())


if __name__ == '__main__':
    unittest.main()
