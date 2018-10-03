import unittest

import tensorflow as tf
import tensorflow_encrypted as tfe
import numpy as np

from tensorflow_encrypted.protocol.pond import PondPublicTensor
from tensorflow_encrypted.tensor.int32 import Int32Factory, Int32Tensor


class TestPondPublicEqual(unittest.TestCase):

    def test_public_compare(self):
        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        expected = np.array([1, 0, 1, 0])
        with tfe.protocol.Pond(tensor_factory=Int32Factory(), use_noninteractive_truncation=True, verify_precision=False, *config.get_players('server0, server1, crypto_producer')) as prot:
            i = Int32Tensor(tf.constant(np.array([100, 200, 100, 300]), dtype=tf.int32))
            input = PondPublicTensor(prot, value_on_0=i, value_on_1=i, is_scaled=False)

            res = prot.equal(input, 100)

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())
                answer = res.eval(sess)
                assert np.array_equal(answer, expected)


if __name__ == '__main__':
    unittest.main()
