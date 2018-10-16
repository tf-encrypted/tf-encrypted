import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

from tf_encrypted.tensor import int100factory, fixed100, fixed100_ni


class TestTruncate(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_interactive_truncate(self):

        prot = tfe.protocol.Pond(
            tensor_factory=int100factory,
            fixedpoint_config=fixed100,
        )

        with tfe.Session() as sess:

            expected = np.array([12345.6789])

            w = prot.define_private_variable(expected * prot.fixedpoint_config.scaling_factor)  # double precision
            v = prot.truncate(w)  # single precision

            sess.run(tf.global_variables_initializer())
            actual = sess.run(v.reveal())

            np.testing.assert_allclose(actual, expected)

    def test_noninteractive_truncate(self):

        prot = tfe.protocol.Pond(
            tensor_factory=int100factory,
            fixedpoint_config=fixed100_ni,
        )

        with tfe.Session() as sess:

            expected = np.array([12345.6789])

            w = prot.define_private_variable(expected * prot.fixedpoint_config.scaling_factor)  # double precision
            v = prot.truncate(w)  # single precision

            sess.run(tf.global_variables_initializer())
            actual = sess.run(v.reveal())

            np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
    unittest.main()
