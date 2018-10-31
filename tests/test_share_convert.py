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

        val_a = np.array([100, -100])
        val_b = np.array([101, 50])

        expected = [201, -49]  # -100 is -99 in L-1

        x_in = prot.define_private_variable(val_a, apply_scaling=False)
        y_in = prot.define_private_variable(val_b, apply_scaling=False)

        x_c = prot.share_convert(x_in)
        y_c = prot.share_convert(y_in)

        result = x_c + y_c

        with tfe.Session() as sess:
            sess.run(tf.global_variables_initializer())

            answer = sess.run(result.reveal().value_on_0.value)
            np.testing.assert_array_equal(answer, expected)


if __name__ == '__main__':
    unittest.main()
