import unittest

import tensorflow as tf
import numpy as np
from tf_encrypted.operations.secure_random import secure_random


class TestSecureRandom(unittest.TestCase):

    def test_wrapper(self):
        expected = [[6610, 5100, 676],
                    [6111, 9801, 5407],
                    [9678, 7188, 8280]]

        with tf.Session():
            output = secure_random([3, 3], maxval=10000).eval()

            np.testing.assert_array_equal(output, expected)

    def test_min_val(self):
        expected = [[3220, 200, -8648],
                    [2223, 9603, 815],
                    [9356, 4377, 6561]]

        with tf.Session():
            output = secure_random([3, 3], minval=-10000, maxval=10000).eval()

            np.testing.assert_array_equal(output, expected)

    def test_invalid_args(self):
        with tf.Session():
            # invalid seed
            with np.testing.assert_raises(ValueError):
                secure_random([3, 3], maxval=10000, seed=[1]).eval()

            # invalid maxval
            with np.testing.assert_raises(ValueError):
                secure_random([3, 3], seed=[1]).eval()

            # invalid dtype
            with np.testing.assert_raises(ValueError):
                secure_random([3, 3], maxval=10000, dtype=tf.float32).eval()

    # def test_rejection(self):
    #     # TODO how to test rejection?!
    #     seed = [87654321, 4321, 8765431, 87654325, 87654321, 874321, 87654321, 87654321]


if __name__ == '__main__':
    unittest.main()
