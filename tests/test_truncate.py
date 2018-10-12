import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe


class TestTruncate(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_interactive_truncate(self):

        prot = tfe.protocol.Pond(
            use_noninteractive_truncation=False
        )

        # TODO[Morten] remove this condition
        if prot.tensor_factory not in [tfe.tensor.int64.int64factory]:

            with tfe.Session() as sess:

                expected = np.array([12345.6789])

                w = prot.define_private_variable(expected * prot.fixedpoint_config.scaling_factor)  # double precision
                v = prot.truncate(w)  # single precision

                sess.run(tf.global_variables_initializer())
                actual = sess.run(v.reveal(), tag='foo')

                np.testing.assert_allclose(actual, expected)

    def test_noninteractive_truncate(self):

        prot = tfe.protocol.Pond(
            use_noninteractive_truncation=True
        )

        with tfe.Session() as sess:

            expected = np.array([12345.6789])

            w = prot.define_private_variable(expected * prot.fixedpoint_config.scaling_factor)  # double precision
            v = prot.truncate(w)  # single precision

            sess.run(tf.global_variables_initializer())
            actual = sess.run(v.reveal(), tag='foo')

            np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
    unittest.main()
