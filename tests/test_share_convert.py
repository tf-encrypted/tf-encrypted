import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
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
        with tfe.protocol.SecureNN(*self.config.get_players('server0, server1, crypto_producer'),
                                   tensor_factory=Int32Factory(),
                                   verify_precision=False) as prot:
            hi = np.array([1, 2, 3, 4])
            x_in = prot.define_private_variable(hi, apply_scaling=False)
            converted = prot.share_convert(x_in)

            with self.config.session() as sess:
                sess.run(tf.global_variables_initializer())

                share0 = Int32Tensor(converted.share0.value)
                share1 = Int32Tensor(converted.share1.value)

                p = PondPrivateTensor(prot, share0, share1, is_scaled=False)

                actual = p.reveal().eval(sess)

                print(actual)


if __name__ == '__main__':
    unittest.main()
