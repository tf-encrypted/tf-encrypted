import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.tensor.prime import PrimeFactory
from tensorflow_encrypted.tensor.int32 import Int32Factory, Int32Tensor
from tensorflow_encrypted.protocol.pond import PondPrivateTensor


class TestShareConvert(unittest.TestCase):

    def setUp(self):
        self.config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

    def test_share_convert(self):

        bit_dtype = PrimeFactory(37)
        val_dtype = Int32Factory()

        with tfe.protocol.SecureNN(*self.config.get_players('server0, server1, crypto_producer'),
                                   tensor_factory=val_dtype,
                                   prime_factory=bit_dtype,
                                   verify_precision=False) as prot:

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
                assert np.array_equal(answer, expected)


if __name__ == '__main__':
    unittest.main()
