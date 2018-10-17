import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.tensor.prime import PrimeFactory
from tf_encrypted.tensor.int100 import int100factory


class TestShare(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

        t = np.array([[1, 2, 3], [4, 5, 6]])
        self.prime_factory = PrimeFactory(67)
        self.int100tensor = int100factory.tensor(t)
        self.primetensor = self.prime_factory.tensor(t)

    def test_share(self):

        with tfe.protocol.Pond(tfe.get_config().get_players('server0, server1, crypto_producer')) as prot:
            shares = prot._share(self.int100tensor)
            out = prot._reconstruct(*shares)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out)

        np.testing.assert_array_equal(final, self.int100tensor.to_native())

    def test_factory_share(self):

        with tfe.protocol.Pond(tfe.get_config().get_players('server0, server1, crypto_producer')) as prot:
            shares = prot._share(self.primetensor)
            out = prot._reconstruct(*shares)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out)

            np.testing.assert_array_equal(final, self.primetensor.value)


if __name__ == '__main__':
    unittest.main()
