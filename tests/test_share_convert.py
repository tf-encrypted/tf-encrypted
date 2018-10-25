import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.tensor.prime import PrimeFactory
from tf_encrypted.tensor import int64factory
from tf_encrypted.protocol.pond import PondPrivateTensor


class TestShareConvert(unittest.TestCase):

    def setUp(self):
        self.config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

    def test_share_convert(self):

        prot = tfe.protocol.SecureNN(
            tensor_factory = int64factory
        )

        bit_dtype = prot.prime_factory
        val_dtype = prot.tensor_factory

        val_a = np.array([1, 2, 3, 4])
        val_b = np.array([1, 2, 3, 4])

        x_in = prot.define_private_variable(val_a, apply_scaling=False)
        y_in = prot.define_private_variable(val_b, apply_scaling=False)

        x_c = prot.share_convert(x_in)
        y_c = prot.share_convert(y_in)

        expected = val_a + val_b
        actual = x_c + y_c

        with tfe.Session() as sess:
            sess.run(tf.global_variables_initializer())

            answer = sess.run(actual.reveal().value_on_0.value)
            print('answer', answer)
            assert np.array_equal(answer, expected)


if __name__ == '__main__':
    unittest.main()
