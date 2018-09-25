import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.tensor.int32 import Int32Factory, Int32Tensor


class TestPrivateCompare(unittest.TestCase):

    def test_privateCompare(self):

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        input = np.array([1, 1, 1, 1]).astype(np.int32)
        r = np.array([2, 0, 2, 0]).astype(np.int32)
        beta = np.array([1, 0, 1, 0]).astype(np.int32)

        # expected = np.array([2, 1, 2, 1]).astype(np.float32)

        with tfe.protocol.SecureNN(*config.get_players('server0, server1, crypto_producer')) as prot:
            #
            input = Int32Tensor(input)
            beta = Int32Tensor(beta)
            r = Int32Tensor(r)

            compare = prot.private_compare(input, r, beta)

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())
                print(f'answer: {compare}')
                # chosen = compare.reveal().eval(sess)
                #
                # assert(np.array_equal(expected, compare))


if __name__ == '__main__':
    unittest.main()
