import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

from tf_encrypted.tensor.int100 import int100factory
from tf_encrypted.tensor.prime import PrimeFactory


class TestLSB(unittest.TestCase):

    def setUp(self):
        # self.M = 2 ** 15 - 1  # this one works
        self.M = 2 ** 15 + 27  # this one doesn't
        # self.M = 2 ** 31 - 3  # this one definitely doesn't
        x = np.random.choice(self.M, (50,))
        # x = np.array([1,2,3,4])
        f_bin = np.vectorize(np.binary_repr)
        f_get = np.vectorize(lambda x, ix: x[ix])
        self.expected_lsb = f_get(f_bin(x), -1).astype(np.int32)
        self.x = x.astype(np.float32)

    def _core_lsb(self, tensor_factory, prime_factory):

        with tfe.protocol.SecureNN(
            tensor_factory=tensor_factory,
            prime_factory=prime_factory,
        ) as prot:

            x_in = prot.define_private_variable(self.x, apply_scaling=False, name='test_lsb_input')
            x_lsb = prot.lsb(x_in)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                lsb = sess.run(x_lsb.reveal(), tag='lsb')

                np.testing.assert_array_equal(self.expected_lsb, lsb)

    def test_lsb(self):
        prime_factory = PrimeFactory(self.M)
        # self._core_lsb(prime_factory, prime_factory)
        self._core_lsb(int100factory, prime_factory)


if __name__ == '__main__':
    unittest.main()
