import unittest
import random

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

from tf_encrypted.tensor import int100factory


class TestLSB(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def _core_lsb(self, tensor_factory, prime_factory):

        f_bin = np.vectorize(np.binary_repr)
        f_get = np.vectorize(lambda x, ix: x[ix])

        raw = np.array([random.randrange(0, 10000000000) for _ in range(20)]).reshape(2, 2, 5)
        expected_lsb = f_get(f_bin(raw), -1).astype(np.int32)

        with tfe.protocol.SecureNN(
            tensor_factory=tensor_factory,
            prime_factory=prime_factory,
        ) as prot:

            x_in = prot.define_private_variable(raw, apply_scaling=False, name='test_lsb_input')
            x_lsb = prot.lsb(x_in)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                actual_lsb = sess.run(x_lsb.reveal(), tag='lsb')

                np.testing.assert_array_equal(actual_lsb, expected_lsb)

    def test_lsb_int100(self):
        self._core_lsb(
            int100factory,
            None
        )


if __name__ == '__main__':
    unittest.main()
