import unittest

import tensorflow as tf
import numpy as np
from tf_encrypted.operations.secure_random import secure_random


class TestSecureRandom(unittest.TestCase):

    def test_wrapper(self):
        expected = [[7738, 4010, 3717], [7182, 302, 6300], [3270, 1433, 6475]]

        with tf.Session():
            output = secure_random([3, 3], maxval=10000).eval()

            np.testing.assert_array_equal(output, expected)

    def test_min_val(self):
        expected = [[7738, 4010, -6283], [-2818, 302, -3700], [-6730, -8567, 6475]]

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


if __name__ == '__main__':
    unittest.main()
