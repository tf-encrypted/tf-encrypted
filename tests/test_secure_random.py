import unittest

import tensorflow as tf
import numpy as np
from tf_encrypted.operations.secure_random import seeded_random_uniform, random_uniform, seed as seed_gen

seed = [87654321, 87654321, 87654321, 87654321, 87654321, 87654321, 87654321, 87654321]


class TestSeededRandom(unittest.TestCase):

    def test_wrapper(self):
        expected = [[6610, 5100, 676],
                    [6111, 9801, 5407],
                    [9678, 7188, 8280]]

        with tf.Session():
            output = seeded_random_uniform([3, 3], seed=seed, maxval=10000).eval()

            np.testing.assert_array_equal(output, expected)

    def test_min_val(self):
        expected = [[3220, 200, -8648],
                    [2223, 9603, 815],
                    [9356, 4377, 6561]]

        with tf.Session():
            output = seeded_random_uniform([3, 3], seed=seed, minval=-10000, maxval=10000).eval()

            np.testing.assert_array_equal(output, expected)

    def test_invalid_args(self):
        with tf.Session():
            # invalid seed
            with np.testing.assert_raises(ValueError):
                seeded_random_uniform([3, 3], maxval=10000, seed=[1]).eval()

            # invalid maxval
            with np.testing.assert_raises(ValueError):
                seeded_random_uniform([3, 3]).eval()

            # invalid dtype
            with np.testing.assert_raises(ValueError):
                seeded_random_uniform([3, 3], seed=seed, maxval=10000, dtype=tf.float32).eval()

    # def test_rejection(self):
    #     # TODO how to test rejection?!
    #     seed = [87654321, 4321, 8765431, 87654325, 87654321, 874321, 87654321, 87654321]


class TestRandomUniform(unittest.TestCase):

    def test_wrapper(self):
        with tf.Session():
            output = random_uniform([3, 3], maxval=10000).eval()

            np.testing.assert_array_equal(output.shape, [3, 3])

    def test_min_val(self):
        with tf.Session():
            output = random_uniform([6], minval=-10000, maxval=0).eval()

            for out in output:
                assert(out < 0)

    def test_invalid_args(self):
        with tf.Session():
            # invalid maxval
            with np.testing.assert_raises(ValueError):
                random_uniform([3, 3]).eval()

            # invalid dtype
            with np.testing.assert_raises(ValueError):
                random_uniform([3, 3], maxval=10000, dtype=tf.float32).eval()


class TestSeed(unittest.TestCase):
    with tf.Session():
        s = seed_gen()

        minval = -2000
        maxval = 0

        shape = [2, 3]

        output = seeded_random_uniform(shape, seed=s, minval=minval, maxval=maxval).eval()

        print(output)

        np.testing.assert_array_equal(output.shape, shape)


if __name__ == '__main__':
    unittest.main()
