# pylint: disable=missing-docstring
import os
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import errors
from tensorflow.python.framework.errors import NotFoundError

import tf_encrypted as tfe

dirname = os.path.dirname(tfe.__file__)
so_name = "{dn}/operations/secure_random/secure_random_module_tf_{tfv}.so"
shared_object = so_name.format(dn=dirname, tfv=tf.__version__)
notfound_msg = "secure_random_module not found"

try:
    secure_random_module = tf.load_op_library(shared_object)
    seeded_random_uniform = secure_random_module.secure_seeded_random_uniform
    random_uniform = secure_random_module.secure_random_uniform
    seed = secure_random_module.secure_seed
except NotFoundError:
    secure_random_module = None


@unittest.skipIf(secure_random_module is None, notfound_msg)
class TestSeededRandomUniform(unittest.TestCase):
    def test_int32_return(self):
        expected = [[749, 945, 451], [537, 795, 111]]

        output = seeded_random_uniform([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], 0, 1000)

        np.testing.assert_array_equal(output, expected)

    def test_int64_return(self):
        expected = [[469, 403, 651], [801, 843, 806]]

        minval = tf.constant(0, dtype=tf.int64)
        maxval = tf.constant(1000, dtype=tf.int64)

        output = seeded_random_uniform([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], minval, maxval)

        np.testing.assert_array_equal(output, expected)

    def test_min_max_range(self):
        minval = tf.constant(-100000000, dtype=tf.int32)
        maxval = tf.constant(100000000, dtype=tf.int32)

        output = seeded_random_uniform(
            [10000], [1, 1, 1, 500, 1, 1, 1, 2], minval, maxval
        )

        for out in output:
            assert -100000000 <= out < 100000000

    def test_invalid_max_min(self):
        minval = tf.constant(1000, dtype=tf.int64)
        maxval = tf.constant(-1000, dtype=tf.int64)

        with np.testing.assert_raises(errors.InvalidArgumentError):
            seeded_random_uniform([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], minval, maxval)

    def test_negative_numbers(self):
        expected = [[-1531, -1597, -1349], [-1199, -1157, -1194]]
        minval = tf.constant(-2000, dtype=tf.int64)
        maxval = tf.constant(-1000, dtype=tf.int64)

        output = seeded_random_uniform([2, 3], [1, 1, 1, 1, 1, 1, 1, 2], minval, maxval)

        np.testing.assert_array_equal(output, expected)
    
    def test_big_tensor(self):
        minval = tf.constant(tf.int64.min, dtype=tf.int64)
        maxval = tf.constant(tf.int64.max, dtype=tf.int64)

        output0 = seeded_random_uniform([20000, 20000], [1, 1, 1, 1, 1, 1, 1, 2], minval, maxval)
        output1 = seeded_random_uniform([20000, 20000], [1, 1, 1, 1, 1, 1, 1, 2], minval, maxval)
        
        np.testing.assert_array_equal(output0, output1)


@unittest.skipIf(secure_random_module is None, notfound_msg)
class TestRandomUniform(unittest.TestCase):
    def test_min_max_range(self):
        minval = tf.constant(-10000000, dtype=tf.int32)
        maxval = tf.constant(10000000, dtype=tf.int32)

        output = random_uniform([1000], minval, maxval)

        for out in output:
            assert -10000000 <= out < 10000000

    def test_small_range(self):
        minval = tf.constant(-10, dtype=tf.int32)
        maxval = tf.constant(10, dtype=tf.int32)

        output = random_uniform([1000], minval, maxval)

        for out in output:
            assert -10 <= out < 10

    def test_neg_range(self):
        minval = tf.constant(-100, dtype=tf.int32)
        maxval = tf.constant(0, dtype=tf.int32)

        output = random_uniform([1000], minval, maxval)

        for out in output:
            assert out < 0


@unittest.skipIf(secure_random_module is None, notfound_msg)
class TestSeed(unittest.TestCase):
    def test_seed(self):
        s = seed()

        minval = tf.constant(-2000, dtype=tf.int64)
        maxval = tf.constant(0, dtype=tf.int64)

        shape = [2, 3]

        output = seeded_random_uniform(shape, s, minval, maxval)

        np.testing.assert_array_equal(output.shape, shape)


if __name__ == "__main__":
    unittest.main()
