import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.tensor.odd_implicit import oddInt32factory


class TestOddImplicitTensor(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def test_add(self) -> None:

        # regular, overflow, underflow
        x = oddInt32factory.tensor(tf.constant([5, 2**31 - 1, -2147483643], dtype=tf.int32))
        y = oddInt32factory.tensor(tf.constant([3, 5, -6], dtype=tf.int32))

        z = x + y

        expected = np.array([8, -2147483643, 2147483646])

        with tf.Session() as sess:
            actual = sess.run(z.value)

        np.testing.assert_array_almost_equal(actual, expected, decimal=3)

    def test_sub(self) -> None:

        # regular, overflow, underflow
        x = oddInt32factory.tensor(tf.constant([10, 6, -6], dtype=tf.int32))
        y = oddInt32factory.tensor(tf.constant([5, -2**31 + 1, -2**31 + 1], dtype=tf.int32))

        z = x - y

        expected = np.array([5, -2147483642, 2147483641])

        with tf.Session() as sess:
            actual = sess.run(z.value)

        np.testing.assert_array_almost_equal(actual, expected, decimal=3)


if __name__ == '__main__':
    unittest.main()
