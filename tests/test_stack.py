import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestStack(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_stack(self):

        with tf.Session() as sess:
            x = tf.constant([1, 4])
            y = tf.constant([2, 5])
            z = tf.constant([3, 6])
            out = tf.stack([x, y, z])

            actual = sess.run(out)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            x = prot.define_private_variable(np.array([1, 4]))
            y = prot.define_private_variable(np.array([2, 5]))
            z = prot.define_private_variable(np.array([3, 6]))

            out = prot.stack((x, y, z))

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out.reveal())

        np.testing.assert_array_equal(final, actual)


if __name__ == '__main__':
    unittest.main()
