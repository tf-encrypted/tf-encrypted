import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.tensor.int32 import Int32Factory


class TestShareConvert(unittest.TestCase):

    def setUp(self):
        self.config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

    def test_share_convert(self):
        with tfe.protocol.SecureNN(*self.config.get_players('server0, server1, crypto_producer'),
                                   tensor_factory=Int32Factory(),
                                   verify_precision=False) as prot:
            hi = np.array([1, 2, 3])
            x_in = prot.define_private_variable(hi, apply_scaling=False)
            converted = prot.share_convert(x_in)

            with self.config.session() as sess:
                sess.run(tf.global_variables_initializer())
                actual = converted.reveal().eval(sess)

                print(actual)


if __name__ == '__main__':
    unittest.main()
