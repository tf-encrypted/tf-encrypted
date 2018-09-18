import unittest

import numpy as np
import tensorflow as tf
from tensorflow_encrypted.tensor.native import NativeTensor


class TestNativeTensor(unittest.TestCase):
    def test_arithmetic(self) -> None:
        x = NativeTensor(tf.constant([2**16, 2**16 + 1]), 2**16)
        y = NativeTensor(tf.constant([2**16 + 2, 2]), 2**16)

        with tf.Session() as sess:
            z = (x * y).value
            z0 = sess.run(z)

            np.testing.assert_array_equal(z0, np.array([0, 2]))

            z = (x + y).value
            z1 = sess.run(z)

            np.testing.assert_array_equal(z1, np.array([2, 3]))

            z = (x - y).value
            z2 = sess.run(z)

            np.testing.assert_array_equal(z2, np.array([65534, 65535]))

    def test_binaraize(self) -> None:
        x = NativeTensor(tf.constant([2, 1000]), 2**16)
        y = x.binarize()

        expected = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        with tf.Session() as sess:
            actual = sess.run(y.value)

        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
