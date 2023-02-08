# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from tf_encrypted.operations import secure_random

SEED = [
    87654321,
    87654321,
    87654321,
    87654321,
    87654321,
    87654321,
    87654321,
    87654321,
]
disabled_msg = "Secure random disabled"
dontskip = secure_random.supports_seeded_randomness()


@unittest.skipUnless(dontskip, disabled_msg)
class TestSeededRandom(unittest.TestCase):
    def test_wrapper(self):
        expected = [
            [7738, 4010, 3717],
            [7182, 302, 6300],
            [3270, 1433, 6475],
        ]

        output = secure_random.seeded_random_uniform(
            [3, 3],
            seed=SEED,
            maxval=10000,
        )

        np.testing.assert_array_equal(output, expected)

    def test_min_val(self):
        expected = [
            [7738, 4010, -6283],
            [-2818, 302, -3700],
            [-6730, -8567, 6475],
        ]

        output = secure_random.seeded_random_uniform(
            [3, 3],
            seed=SEED,
            minval=-10000,
            maxval=10000,
        )

        np.testing.assert_array_equal(output, expected)

    def test_invalid_args(self):
        # invalid seed
        with np.testing.assert_raises(InvalidArgumentError):
            secure_random.seeded_random_uniform(
                [3, 3],
                maxval=10000,
                seed=[1],
            )

        # invalid maxval
        with np.testing.assert_raises(ValueError):
            secure_random.seeded_random_uniform([3, 3])

        # invalid dtype
        with np.testing.assert_raises(ValueError):
            secure_random.seeded_random_uniform(
                [3, 3],
                seed=SEED,
                maxval=10000,
                dtype=tf.float32,
            )

    def test_rejection(self):
        m = 1129

        seed0 = [
            2108217960,
            -1340439062,
            476173466,
            -681389164,
            -1502583120,
            1663373136,
            2144760032,
            -1591917499,
        ]

        out0 = secure_random.seeded_random_uniform(
            [64, 4500],
            seed=seed0,
            maxval=m,
            dtype=tf.int32,
        )
        out1 = secure_random.seeded_random_uniform(
            [64, 4500],
            seed=seed0,
            maxval=m,
            dtype=tf.int32,
        )

        np.testing.assert_array_equal(out0, out1)


@unittest.skipUnless(dontskip, disabled_msg)
class TestRandomUniform(unittest.TestCase):
    def test_wrapper(self):
        output = secure_random.random_uniform([3, 3], maxval=10000)
        np.testing.assert_array_equal(output.shape, [3, 3])

    def test_min_val(self):
        output = secure_random.random_uniform([6], minval=-10000, maxval=0)
        for out in output:
            assert out < 0

    def test_invalid_args(self):
        # invalid maxval
        with np.testing.assert_raises(ValueError):
            secure_random.random_uniform([3, 3])

        # invalid dtype
        with np.testing.assert_raises(ValueError):
            secure_random.random_uniform(
                [3, 3],
                maxval=10000,
                dtype=tf.float32,
            )


@unittest.skipUnless(dontskip, disabled_msg)
class TestSeed(unittest.TestCase):
    def test_seed_generation(self):
        s = secure_random.secure_seed()

        minval = -2000
        maxval = 0

        shape = [2, 3]

        output = secure_random.seeded_random_uniform(
            shape,
            seed=s,
            minval=minval,
            maxval=maxval,
        )

        np.testing.assert_array_equal(output.shape, shape)


if __name__ == "__main__":
    unittest.main()
