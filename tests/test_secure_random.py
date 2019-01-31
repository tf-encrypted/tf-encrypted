import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.operations import secure_random


SEED = [87654321, 87654321, 87654321, 87654321, 87654321, 87654321, 87654321, 87654321]


@unittest.skipUnless(secure_random.supports_seeded_randomness(), "Secure random disabled")
class TestSeededRandom(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def tearDown(self):
        tf.reset_default_graph()

    def test_wrapper(self):
        expected = [[6610, 5100, 676],
                    [6111, 9801, 5407],
                    [9678, 7188, 8280]]

        with tf.Session():
            output = secure_random.seeded_random_uniform([3, 3], seed=SEED, maxval=10000).eval()

            np.testing.assert_array_equal(output, expected)

    def test_min_val(self):
        expected = [[3220, 200, -8648],
                    [2223, 9603, 815],
                    [9356, 4377, 6561]]

        with tf.Session():
            output = secure_random.seeded_random_uniform([3, 3], seed=SEED, minval=-10000, maxval=10000).eval()

            np.testing.assert_array_equal(output, expected)

    def test_invalid_args(self):
        with tf.Session():
            # invalid seed
            with np.testing.assert_raises(ValueError):
                secure_random.seeded_random_uniform([3, 3], maxval=10000, seed=[1]).eval()

            # invalid maxval
            with np.testing.assert_raises(ValueError):
                secure_random.seeded_random_uniform([3, 3]).eval()

            # invalid dtype
            with np.testing.assert_raises(ValueError):
                secure_random.seeded_random_uniform([3, 3], seed=SEED, maxval=10000, dtype=tf.float32).eval()

    def test_rejection(self):
        m = 1129

        seed0 = [2108217960, -1340439062, 476173466, -681389164, -1502583120, 1663373136, 2144760032, -1591917499]

        with tf.Session():
            out0 = secure_random.seeded_random_uniform([64, 4500], seed=seed0, maxval=m, dtype=tf.int32).eval()
            out1 = secure_random.seeded_random_uniform([64, 4500], seed=seed0, maxval=m, dtype=tf.int32).eval()

            np.testing.assert_array_equal(out0, out1)


@unittest.skipUnless(secure_random.supports_secure_randomness(), "Secure random disabled")
class TestRandomUniform(unittest.TestCase):

    def test_wrapper(self):
        with tf.Session():
            output = secure_random.random_uniform([3, 3], maxval=10000).eval()

            np.testing.assert_array_equal(output.shape, [3, 3])

    def test_min_val(self):
        with tf.Session():
            output = secure_random.random_uniform([6], minval=-10000, maxval=0).eval()

            for out in output:
                assert(out < 0)

    def test_invalid_args(self):
        with tf.Session():
            # invalid maxval
            with np.testing.assert_raises(ValueError):
                secure_random.random_uniform([3, 3]).eval()

            # invalid dtype
            with np.testing.assert_raises(ValueError):
                secure_random.random_uniform([3, 3], maxval=10000, dtype=tf.float32).eval()


@unittest.skipUnless(secure_random.supports_seeded_randomness(), "Secure random disabled")
class TestSeed(unittest.TestCase):

    def test_seed_generation(self):
        with tf.Session():
            s = secure_random.seed()

            minval = -2000
            maxval = 0

            shape = [2, 3]

            output = secure_random.seeded_random_uniform(shape, seed=s, minval=minval, maxval=maxval).eval()

            np.testing.assert_array_equal(output.shape, shape)


if __name__ == '__main__':
    unittest.main()
