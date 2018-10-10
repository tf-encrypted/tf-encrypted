import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.tensor.prime import PrimeFactory


class TestLSB(unittest.TestCase):

    def setUp(self):
        # self.M = 2 ** 15 - 1  # this one works
        self.M = 2 ** 15 + 27  # this one doesn't
        # self.M = 2 ** 31 - 3  # this one definitely doesn't
        x = np.random.choice(self.M, (10,))
        # x = np.array([1,2,3,4])
        f_bin = np.vectorize(np.binary_repr)
        f_get = np.vectorize(lambda x, ix: x[ix])
        self.expected_lsb = f_get(f_bin(x), -1).astype(np.int32)
        self.x = x.astype(np.float32)

    def _core_lsb(self, factory):
        with tfe.protocol.SecureNN(
            tensor_factory=factory,
            verify_precision=False
        ) as prot:

            x_in = prot.define_private_variable(self.x, apply_scaling=False, name='test_lsb_input')
            x_lsb = prot.lsb(x_in)
            x_lsb.share0.value = tf.Print(x_lsb.share0.value, [x_lsb.reveal().value_on_0.value], 'lsbo', summarize=10)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                lsb = sess.run(x_lsb.reveal().value_on_0.value)

                np.testing.assert_array_equal(self.expected_lsb, lsb)

    def test_lsb(self):
        self._core_lsb(PrimeFactory(self.M))


if __name__ == '__main__':
    unittest.main()
