import unittest
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import errors
from tensorflow.python.framework.errors import NotFoundError

import tf_encrypted as tfe


dirname = os.path.dirname(tfe.__file__)
shared_object = dirname + '/operations/secure_random/secure_random_module_tf_' + tf.__version__ + '.so'

try:
    secure_random_module = tf.load_op_library(shared_object)
    seeded_random_uniform = secure_random_module.secure_seeded_random_uniform
    random_uniform = secure_random_module.secure_random_uniform
    seed = secure_random_module.secure_seed
except NotFoundError:
    secure_random_module = None


@unittest.skipIf(secure_random_module is None, "secure_random_module not found")
class TestSeededRandomUniform(unittest.TestCase):

    def test_int32_return(self):
        expected = [[608, 425, 925], [198, 891, 721]]

        with tf.Session():
            output = seeded_random_uniform([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], 0, 1000).eval()

            np.testing.assert_array_equal(output, expected)

    def test_int64_return(self):
        expected = [[425, 198, 721], [911, 617, 113]]

        with tf.Session():
            minval = tf.constant(0, dtype=tf.int64)
            maxval = tf.constant(1000, dtype=tf.int64)

            output = seeded_random_uniform([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], minval, maxval).eval()

            np.testing.assert_array_equal(output, expected)

    def test_min_max_range(self):
        with tf.Session():
            minval = tf.constant(-100000000, dtype=tf.int32)
            maxval = tf.constant(100000000, dtype=tf.int32)

            output = seeded_random_uniform([10000], [1, 1, 1, 500, 1, 1, 1, 2], minval, maxval).eval()

            for out in output:
                assert(out >= -100000000 and out < 100000000)

    def test_invalid_max_min(self):
        with tf.Session():
            minval = tf.constant(1000, dtype=tf.int64)
            maxval = tf.constant(-1000, dtype=tf.int64)

            with np.testing.assert_raises(errors.InvalidArgumentError):
                seeded_random_uniform([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], minval, maxval).eval()

    def test_negative_numbers(self):
        expected = [[-1575, -1802, -1279], [-1089, -1383, -1887]]
        with tf.Session():
            minval = tf.constant(-2000, dtype=tf.int64)
            maxval = tf.constant(-1000, dtype=tf.int64)

            output = seeded_random_uniform([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], minval, maxval).eval()

            np.testing.assert_array_equal(output, expected)


@unittest.skipIf(secure_random_module is None, "secure_random_module not found")
class TestRandomUniform(unittest.TestCase):
    def test_min_max_range(self):
        with tf.Session():
            minval = tf.constant(-10000000, dtype=tf.int32)
            maxval = tf.constant(10000000, dtype=tf.int32)

            output = random_uniform([1000], minval, maxval).eval()

            for out in output:
                assert(out >= -10000000 and out < 10000000)

    def test_small_range(self):
        with tf.Session():
            minval = tf.constant(-10, dtype=tf.int32)
            maxval = tf.constant(10, dtype=tf.int32)

            output = random_uniform([1000], minval, maxval).eval()

            for out in output:
                assert(out >= -10 and out < 10)

    def test_neg_range(self):
        with tf.Session():
            minval = tf.constant(-100, dtype=tf.int32)
            maxval = tf.constant(0, dtype=tf.int32)

            output = random_uniform([1000], minval, maxval).eval()

            for out in output:
                assert(out < 0)


@unittest.skipIf(secure_random_module is None, "secure_random_module not found")
class TestSeed(unittest.TestCase):
    def test_seed(self):
        with tf.Session():
            s = seed()

            minval = tf.constant(-2000, dtype=tf.int64)
            maxval = tf.constant(0, dtype=tf.int64)

            shape = [2, 3]

            output = seeded_random_uniform(shape, s, minval, maxval).eval()

            np.testing.assert_array_equal(output.shape, shape)


if __name__ == '__main__':
    unittest.main()
