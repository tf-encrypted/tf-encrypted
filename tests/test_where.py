import unittest

import tensorflow as tf
import tensorflow_encrypted as tfe
import numpy as np

from tensorflow_encrypted.protocol.pond import PondPrivateTensor, PondPublicTensor
from tensorflow_encrypted.tensor.int32 import Int32Factory, Int32Tensor
from tensorflow_encrypted.tensor.prime import prime_factory


class TestPondPublicEqual(unittest.TestCase):

    def test_public_compare(self):
        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        expected = np.array([[0], [2]])
        with tfe.protocol.Pond(tensor_factory=Int32Factory(), use_noninteractive_truncation=True, verify_precision=False, *config.get_players('server0, server1, crypto_producer')) as prot:
            i = Int32Tensor(tf.constant(np.array([100, 200, 100, 300]), dtype=tf.int32))
            input = PondPublicTensor(prot, value_on_0=i, value_on_1=i, is_scaled=False)

            eq = prot.equal(input, 100)
            where = prot.where(eq)

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())
                answer = where.eval(sess)
                assert np.array_equal(answer, expected)


if __name__ == '__main__':
    unittest.main()
