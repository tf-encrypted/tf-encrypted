import unittest

import numpy as np
import tensorflow as tf
from tensorflow_encrypted.tensor.odd_implicit import OddImplicitTensor


class TestOddImplicitTensor(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    # TODO these tests are very robust
    def test_add(self) -> None:
        modulus = 2**32

        x = OddImplicitTensor(tf.constant([2 ** 32 - 1, 2 ** 32 - 2], dtype=tf.int32), modulus)
        y = OddImplicitTensor(tf.constant([2, 3], dtype=tf.int32), modulus)

        z = x + y

        expected = np.array([2, 2])

        with tf.Session() as sess:
            actual = sess.run(z.value)

            np.testing.assert_array_almost_equal(actual, expected, decimal=3)

    def test_sub(self) -> None:
        modulus = 2**32

        x = OddImplicitTensor(tf.constant([2 ** 32 + 300], dtype=tf.int32), modulus)
        y = OddImplicitTensor(tf.constant([200], dtype=tf.int32), modulus)

        z = x - y

        expected = np.array([101])

        with tf.Session() as sess:
            actual = sess.run(z.value)

            np.testing.assert_array_almost_equal(actual, expected, decimal=3)


if __name__ == '__main__':
    unittest.main()
