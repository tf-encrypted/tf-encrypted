import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class Testconcat(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_concat(self):

        with tf.Session() as sess:
            t1 = [[1, 2, 3], [4, 5, 6]]
            t2 = [[7, 8, 9], [10, 11, 12]]
            out = tf.concat([t1, t2], 0)
            actual = sess.run(out)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            x = prot.define_private_variable(np.array(t1))
            y = prot.define_private_variable(np.array(t2))

            out = prot.concat([x, y], 0)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out.reveal())

        np.testing.assert_array_equal(final, actual)

    def test_masked_concat(self):

        with tf.Session() as sess:
            t1 = [[1, 2, 3], [4, 5, 6]]
            t2 = [[7, 8, 9], [10, 11, 12]]
            out = tf.concat([t1, t2], 0)
            actual = sess.run(out)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            x = prot.mask(prot.define_private_variable(np.array(t1)))
            y = prot.mask(prot.define_private_variable(np.array(t2)))

            out = prot.concat([x, y], 0)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out.unmasked.reveal())

        np.testing.assert_array_equal(final, actual)


if __name__ == '__main__':
    unittest.main()
