# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.protocol.securenn.odd_tensor import oddint64_factory


class TestOddImplicitTensor(unittest.TestCase):
    def test_tensor(self) -> None:
        # regular, overflow, underflow
        x = oddint64_factory.tensor(tf.constant([-2, -1, 0, 1], dtype=tf.int64))
        expected = np.array([-2, 0, 0, 1])
        actual = x.value
        np.testing.assert_array_almost_equal(actual, expected, decimal=3)

    def test_add(self) -> None:

        # regular, overflow, underflow
        x = oddint64_factory.tensor(tf.constant([2, -2], dtype=tf.int64))
        y = oddint64_factory.tensor(tf.constant([3, 3], dtype=tf.int64))

        z = x + y

        expected = np.array([5, 2])
        actual = z.value
        np.testing.assert_array_almost_equal(actual, expected, decimal=3)

    def test_sub(self) -> None:

        # regular, overflow, underflow
        x = oddint64_factory.tensor(tf.constant([2, -2], dtype=tf.int64))
        y = oddint64_factory.tensor(tf.constant([3, 3], dtype=tf.int64))

        z = x - y

        expected = np.array([-2, -5])
        actual = z.value
        np.testing.assert_array_almost_equal(actual, expected, decimal=3)


if __name__ == "__main__":
    unittest.main()
