import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe


class TestTruncate(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_interactive_truncate(self):

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        server0, server1, cp = config.get_players('server0, server1, crypto_producer')

        prot = tfe.protocol.Pond(
            server0, server1, cp,
            use_interactive_truncation=True
        )

        with config.session() as sess:

            expected = np.array([12345.6789])

            w = prot.define_private_variable(expected * (2**16))  # double precision
            v = prot.truncate(w)  # single precision

            sess.run(tf.global_variables_initializer())
            actual = v.reveal().eval(sess, tag='foo')

            assert np.isclose(actual, expected).all(), actual

    def test_noninteractive_truncate(self):

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        server0, server1, cp = config.get_players('server0, server1, crypto_producer')

        prot = tfe.protocol.Pond(
            server0, server1, cp,
            use_interactive_truncation=False
        )

        with config.session() as sess:

            expected = np.array([12345.6789])

            w = prot.define_private_variable(expected * (2**16))  # double precision
            v = prot.truncate(w)  # single precision

            sess.run(tf.global_variables_initializer())
            actual = v.reveal().eval(sess, tag='foo')

            assert np.isclose(actual, expected).all(), actual


if __name__ == '__main__':
    unittest.main()
