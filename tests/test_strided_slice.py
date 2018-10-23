import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestStridedSlice(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_strided_slice(self):

        with tf.Session() as sess:
            t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                             [[3, 3, 3], [4, 4, 4]],
                             [[5, 5, 5], [6, 6, 6]]])
            out = tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])

            actual = sess.run(out)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            x = np.array([[[1, 1, 1], [2, 2, 2]],
                          [[3, 3, 3], [4, 4, 4]],
                          [[5, 5, 5], [6, 6, 6]]])

            out = prot.define_private_variable(x)

            out = prot.strided_slice(out, [1, 0, 0], [2, 1, 3], [1, 1, 1])

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out.reveal())

        np.testing.assert_array_equal(final, actual)


if __name__ == '__main__':
    unittest.main()
