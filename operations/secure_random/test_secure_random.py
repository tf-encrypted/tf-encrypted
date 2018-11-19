import unittest

import tensorflow as tf
from tensorflow.python.framework import errors
import numpy as np
import tf_encrypted as tfe
import os

dirname = os.path.dirname(tfe.__file__)
shared_object = dirname + '/operations/secure_random/secure_random_module.so'
secure_random_module = tf.load_op_library(shared_object)


class TestSecureRandom(unittest.TestCase):

    def test_int32_return(self):
        expected = [[749, 945, 451], [537, 795, 111]]

        with tf.Session():
            output = secure_random_module.secure_random([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], 0, 1000).eval()

            np.testing.assert_array_equal(output, expected)

    def test_int64_return(self):
        expected = [[469, 403, 651], [801, 843, 806]]

        with tf.Session():
            minval = tf.constant(0, dtype=tf.int64)
            maxval = tf.constant(1000, dtype=tf.int64)

            output = secure_random_module.secure_random([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], minval, maxval).eval()

            np.testing.assert_array_equal(output, expected)

    def test_min_max_range(self):
        with tf.Session():
            minval = tf.constant(-100000000, dtype=tf.int32)
            maxval = tf.constant(100000000, dtype=tf.int32)

            output = secure_random_module.secure_random([1000], [1, 1, 1, 500, 1, 1, 1, 2], minval, maxval).eval()

            for out in output:
                assert(out >= -100000000 and out < 100000000)

    def test_invalid_max_min(self):
        with tf.Session():
            minval = tf.constant(1000, dtype=tf.int64)
            maxval = tf.constant(-1000, dtype=tf.int64)

            with np.testing.assert_raises(errors.InvalidArgumentError):
                secure_random_module.secure_random([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], minval, maxval).eval()

    def test_negative_numbers(self):
        expected = [[-1531, -1597, -1349], [-1199, -1157, -1194]]

        with tf.Session():
            minval = tf.constant(-2000, dtype=tf.int64)
            maxval = tf.constant(-1000, dtype=tf.int64)

            output = secure_random_module.secure_random([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], minval, maxval).eval()

            np.testing.assert_array_equal(output, expected)


if __name__ == '__main__':
    unittest.main()
