import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.tensor.prime import prime_factory
from tensorflow_encrypted.tensor.int100 import Int100Factory


class TestShare(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])
        t = np.array([[1, 2, 3], [4, 5, 6]])
        int100factory = Int100Factory()
        self.prime_factory = prime_factory(67)
        self.int100tensor = int100factory.Tensor.from_native(t)
        self.primetensor = self.prime_factory.Tensor.from_native(t)

    def test_share(self):

        with tfe.protocol.Pond(*self.config.get_players('server0, server1, crypto_producer')) as prot:
            shares = prot._share(self.int100tensor)
            out = prot._reconstruct(*shares)

            with self.config.session() as sess:
                sess.run(tf.global_variables_initializer())
                final = out.eval(sess)

        np.testing.assert_array_equal(final.to_native(), self.int100tensor.to_native())

    def test_factory_share(self):

        with tfe.protocol.Pond(*self.config.get_players('server0, server1, crypto_producer')) as prot:
            shares = prot._share(self.primetensor, factory=self.prime_factory)
            out = prot._reconstruct(*shares)

            with self.config.session() as sess:
                sess.run(tf.global_variables_initializer())
                final = out.eval(sess)

            np.testing.assert_array_equal(final.value, self.primetensor.value)


if __name__ == '__main__':
    unittest.main()
