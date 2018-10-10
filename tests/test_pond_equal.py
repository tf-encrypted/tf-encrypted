import unittest

import tensorflow as tf
import tensorflow_encrypted as tfe
import numpy as np

from tensorflow_encrypted.protocol.pond import PondPublicTensor
from tensorflow_encrypted.tensor.int32 import int32factory


class TestPondPublicEqual(unittest.TestCase):

    def test_public_compare(self):

        expected = np.array([1, 0, 1, 0])

        with tfe.protocol.Pond(
            tensor_factory=int32factory,
            use_noninteractive_truncation=True,
            verify_precision=False
        ) as prot:

            x_raw = int32factory.constant(np.array([100, 200, 100, 300]))
            x = PondPublicTensor(prot, value_on_0=x_raw, value_on_1=x_raw, is_scaled=False)

            res = prot.equal(x, 100)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                answer = sess.run(res)

            assert np.array_equal(answer, expected)


if __name__ == '__main__':
    unittest.main()
