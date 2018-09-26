import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.tensor.prime import prime_factory


class TestLSB(unittest.TestCase):

    def setUp(self):
        self.config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])
        self.M = 2 ** 31
        x = np.random.choice(self.M, (10,))
        f_bin = np.vectorize(np.binary_repr)
        f_get = np.vectorize(lambda x, ix: x[ix])
        self.expected_lsb = f_get(f_bin(x), -1).astype(np.int32)
        self.x = x.astype(np.float32)

    def _core_lsb(self, factory):
        with tfe.protocol.SecureNN(*self.config.get_players('server0, server1, crypto_producer'),
                                   tensor_factory=factory,
                                   verify_precision=False) as prot:
            x_in = prot.define_private_variable(self.x, apply_scaling=False)
            x_lsb = prot.lsb(x_in)

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())
                lsb = x_lsb.reveal().eval(sess)

                assert(np.array_equal(self.expected_lsb, lsb))

    def test_lsb(self):
        self._core_lsb(prime_factory(self.M))


if __name__ == '__main__':
    unittest.main()
