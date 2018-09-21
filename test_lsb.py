import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.tensor.int100 import M


class TestLSB(unittest.TestCase):

    def test_lsb(self):

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        x = np.random.choice(2**31, (10,))
        f_bin = np.vectorize(np.binary_repr)
        f_get = np.vectorize(lambda x, ix: x[ix])
        expected_lsb = f_get(f_bin(x), -1).astype(np.int32)

        with tfe.protocol.SecureNN(*config.get_players('server0, server1, crypto_producer')) as prot:
            x_in = prot.define_private_variable(x.astype(np.float32))
            x_lsb = prot.lsb(x_in)

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())
                lsb = x_lsb.reveal().eval(sess)

                assert(np.array_equal(expected_lsb, lsb))


if __name__ == '__main__':
    unittest.main()
