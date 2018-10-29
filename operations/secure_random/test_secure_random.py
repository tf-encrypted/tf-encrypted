import unittest

import tensorflow as tf
import numpy as np
import tf_encrypted as tfe
import os

dirname = os.path.dirname(tfe.__file__)
shared_object = dirname + '/operations/secure_random/secure_random.so'


class TestSecureRandom(unittest.TestCase):

    def test_int32_return(self):
        secure_random_module = tf.load_op_library(shared_object)

        expected = [[749, 945, 451], [537, 795, 111]]

        with tf.Session():
            output = secure_random_module.secure_random([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], 0, 1000).eval()

            np.testing.assert_array_equal(output, expected)

    def test_int64_return(self):
        secure_random_module = tf.load_op_library(shared_object)

        expected = [[469, 403, 651], [801, 843, 806]]

        with tf.Session():
            minval = tf.constant(0, dtype=tf.int64)
            maxval = tf.constant(1000, dtype=tf.int64)

            output = secure_random_module.secure_random([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], minval, maxval).eval()

            np.testing.assert_array_equal(output, expected)


if __name__ == '__main__':
    unittest.main()
