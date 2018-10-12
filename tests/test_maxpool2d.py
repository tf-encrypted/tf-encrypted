import unittest
import tensorflow_encrypted as tfe
import tensorflow as tf
import numpy as np
from tensorflow_encrypted.protocol.pond import PondPublicTensor
from tensorflow_encrypted.layers.pooling import MaxPooling2D


class TestMaxPooling2D(unittest.TestCase):
    def test_maxpool2d(self):
        prot = tfe.protocol.SecureNN(
            use_noninteractive_truncation=True,
            verify_precision=False
        )

        tfe.set_protocol(prot)

        input = np.array([[[[1, 2, 3, 4],
                            [3, 2, 4, 1],
                            [1, 2, 3, 4],
                            [3, 2, 4, 1]]]])

        expected = np.array([[[[3, 4],
                              [3, 4]]]], dtype=np.float64)

        input = tfe.define_private_variable(input)
        pool = MaxPooling2D([0, 1, 4, 4], pool_size=2, padding="VALID")
        result = pool.forward(input)

        with tfe.Session() as sess:
            sess.run(tf.global_variables_initializer())
            answer = sess.run(result.reveal())

        assert np.array_equal(answer, expected)


if __name__ == '__main__':
    unittest.main()
