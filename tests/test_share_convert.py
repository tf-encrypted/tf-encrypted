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
            tensor_factory=int64factory
        )

        bit_dtype = prot.prime_factory
        val_dtype = prot.tensor_factory

        val_a = np.array([100])

        x_in = prot.define_private_variable(val_a, apply_scaling=False)

        x_c = prot.share_convert_2(x_in)

        with tfe.Session() as sess:
            sess.run(tf.global_variables_initializer())

            answer = sess.run(x_c.reveal().value_on_0.value)
            print('answer', answer)


if __name__ == '__main__':
    unittest.main()
