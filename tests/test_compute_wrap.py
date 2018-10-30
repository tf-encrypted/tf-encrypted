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

    def test_compute_wrap(self):
        prot = tfe.protocol.SecureNN(
            tensor_factory=int64factory
        )

        bit_dtype = prot.prime_factory
        val_dtype = prot.tensor_factory

        x = prot.define_public_variable(np.array([
            0,
            -1,
            9223372036854775807,
            -9223372036854775808
        ]))

        y = prot.define_public_variable(np.array([
            -1,
            1,
            1,
            -1
        ]))

        z = x.value_on_0.compute_wrap(y.value_on_0, prot.tensor_factory.modulus)

        with tfe.Session() as sess:
            sess.run(tf.global_variables_initializer())
            answer = sess.run([z.value, (x+y).value_on_0.value])

            print('answer', answer)

    # def test_compute_wrap(self):
    #     prot = tfe.protocol.SecureNN(
    #         tensor_factory=int64factory
    #     )
    #
    #     bit_dtype = prot.prime_factory
    #     val_dtype = prot.tensor_factory
    #
    #     x = prot.define_public_variable(np.array([
    #         tf.int64.max,
    #         tf.int64.max-1,
    #         tf.int64.max-1,
    #         tf.int64.max,
    #         tf.int64.min
    #     ]))
    #
    #     y = prot.define_public_variable(np.array([
    #         1,
    #         1,
    #         2,
    #         0,
    #         -1
    #     ]))
    #
    #     z = x.value_on_0.compute_wrap(y.value_on_0, prot.tensor_factory.modulus)
    #
    #     with tfe.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         answer = sess.run([z.value, (x+y).value_on_0.value])
    #
    #         print('answer', answer)


if __name__ == '__main__':
    unittest.main()
