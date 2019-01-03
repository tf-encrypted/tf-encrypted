import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.protocol.securenn.odd_tensor import oddInt64factory


class TestOddImplicitTensor(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def test_add(self) -> None:

        # regular, overflow, underflow
        x = oddInt64factory.tensor(tf.constant([2, -2], dtype=tf.int64))
        y = oddInt64factory.tensor(tf.constant([3, 3], dtype=tf.int64))

        z = x + y

        expected = np.array([5, 2])

        with tf.Session() as sess:
            actual = sess.run(z.value)

        np.testing.assert_array_almost_equal(actual, expected, decimal=3)

    def test_sub(self) -> None:

        # regular, overflow, underflow
        x = oddInt64factory.tensor(tf.constant([2, -2], dtype=tf.int64))
        y = oddInt64factory.tensor(tf.constant([3, 3], dtype=tf.int64))

        z = x - y

        expected = np.array([-2, -5])

        with tf.Session() as sess:
            actual = sess.run(z.value)

        np.testing.assert_array_almost_equal(actual, expected, decimal=3)


if __name__ == '__main__':
    unittest.main()
