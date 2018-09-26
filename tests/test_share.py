import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.tensor.prime import prime_factory


class TestShare(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.config = config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])
        self.tensor = [[1, 2, 3], [4, 5, 6]]


    def test_share(self):

        with tfe.protocol.Pond(*self.config.get_players('server0, server1, crypto_producer')) as prot:
            shares = prot._share(np.array(self.tensor))
            out = prot._reconstruct(shares)

            with self.config.session() as sess:
                sess.run(tf.global_variables_initializer())
                final = out.eval(sess)

        np.testing.assert_array_equal(final, self.tensor)

    def test_factory_share(self):

        with tfe.protocol.Pond(*self.config.get_players('server0, server1, crypto_producer')) as prot:
            shares = prot._share(np.array(self.tensor), factory=prime_factory(67))
            out = prot._reconstruct(shares)

            with self.config.session() as sess:
                sess.run(tf.global_variables_initializer())
                final = out.eval(sess)

            np.testing.assert_array_equal(final, self.tensor)
