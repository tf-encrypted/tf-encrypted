import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.tensor import int64factory


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

        x = np.arange(0, tf.int64.max, 9223372036854776)  # 1000 entries
        expected = np.arange(0, tf.int64.max, 9223372036854776)

        x_in = prot.define_private_variable(x, apply_scaling=False)
        x_convert = prot.share_convert(x_in)

        with tfe.Session() as sess:
            sess.run(tf.global_variables_initializer())
            answer = sess.run(x_convert.reveal().value_on_0.value)
            print(answer, expected)
            # np.testing.assert_array_equal(answer, expected)


if __name__ == '__main__':
    unittest.main()
