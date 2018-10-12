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

            hi = np.array([1, 2, 3, 4])
            x_in = prot.define_private_variable(hi, apply_scaling=False)
            converted = prot.share_convert(x_in)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())

                share0 = Int32Tensor(converted.share0.value)
                share1 = Int32Tensor(converted.share1.value)

                p = PondPrivateTensor(prot, share0, share1, is_scaled=False)

                actual = sess.run(p.reveal().value_on_0.value)
                # actual = p.reveal().eval(sess)

                print(actual)


if __name__ == '__main__':
    unittest.main()
