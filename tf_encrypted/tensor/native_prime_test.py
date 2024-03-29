# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.tensor import native_factory


class TestPrimeTensor(unittest.TestCase):
    def set_up_indexing(self):
        m = 2**31

        prime_factory = native_factory(tf.int32, m)

        self.np_fix1d = np.arange(24)
        self.np_fix2d = self.np_fix1d.reshape(8, 3)
        self.np_fix3d = self.np_fix1d.reshape(2, 4, 3)
        self.np_fix4d = self.np_fix1d.reshape(2, 2, 2, 3)
        self.prime_fix1d = prime_factory.tensor(self.np_fix1d)
        self.prime_fix2d = prime_factory.tensor(self.np_fix2d)
        self.prime_fix3d = prime_factory.tensor(self.np_fix3d)
        self.prime_fix4d = prime_factory.tensor(self.np_fix4d)
        self.np_fixtures = [getattr(self, "np_fix{}d".format(i)) for i in range(1, 5)]
        self.prime_fixtures = [
            getattr(self, "prime_fix{}d".format(i)) for i in range(1, 5)
        ]

    @unittest.skip
    def test_basic_indexing(self):
        self.set_up_indexing()
        for np_fix, prime_fix in zip(self.np_fixtures, self.prime_fixtures):
            n = len(np_fix.shape)
            for filler in [0, 1, -1]:
                ixs = [filler for _ in range(n)]
                np.testing.assert_equal(np_fix[ixs], prime_fix[ixs].value)

    @unittest.skip
    def test_slice_indexing(self):
        self.set_up_indexing()
        for np_fix, prime_fix in zip(self.np_fixtures, self.prime_fixtures):
            ndim = len(np_fix.shape)
            if ndim == 1:
                np.testing.assert_equal(
                    np_fix[2:5],
                    prime_fix[2:5].value,
                )
                continue
            np.testing.assert_equal(
                np_fix[:, 0],
                prime_fix[:, 0].value,
            )
            np.testing.assert_equal(
                np_fix[:, 1],
                prime_fix[:, 1].value,
            )
            np.testing.assert_equal(
                np_fix[:, -1],
                prime_fix[:, -1].value,
            )
            if ndim > 2:
                np.testing.assert_equal(
                    np_fix[:, :-1, ...],
                    prime_fix[:, :-1, ...].value,
                )
                np.testing.assert_equal(
                    np_fix[:, :1, ...],
                    prime_fix[:, :1, ...].value,
                )
                np.testing.assert_equal(
                    np_fix[:, 1:, ...],
                    prime_fix[:, 1:, ...].value,
                )
            elif ndim == 2:
                np.testing.assert_equal(
                    np_fix[:, :2],
                    prime_fix[:, :-1].value,
                )
                np.testing.assert_equal(
                    np_fix[:, 1:],
                    prime_fix[:, 1:].value,
                )

    @unittest.skip
    def test_ellipsis_indexing(self):
        self.set_up_indexing()
        for np_fix, prime_fix in zip(self.np_fixtures, self.prime_fixtures):
            np.testing.assert_equal(
                np_fix[0, ...],
                prime_fix[0, ...].value,
            )
            np.testing.assert_equal(
                np_fix[1, ...],
                prime_fix[1, ...].value,
            )
            np.testing.assert_equal(
                np_fix[..., -1],
                prime_fix[..., -1].value,
            )

    def test_arithmetic(self) -> None:
        prime_factory = native_factory(tf.int32, 2**16)

        x = prime_factory.tensor(tf.constant([2**16, 2**16 + 1]))
        y = prime_factory.tensor(tf.constant([2**16 + 2, 2]))

        z0 = (x * y).value
        np.testing.assert_array_equal(z0, np.array([0, 2]))

        z1 = (x + y).value
        np.testing.assert_array_equal(z1, np.array([2, 3]))

        z2 = (x - y).value
        np.testing.assert_array_equal(z2, np.array([65534, 65535]))

    def test_binarize(self) -> None:
        prime_factory = native_factory(tf.int32, 1001)

        x = prime_factory.tensor(
            tf.constant(
                [3, -1, 0], shape=[3], dtype=np.int32  # == 3  # == p-1 == max  # min
            )
        )

        y = x.bits()

        expected = np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ).reshape([3, 10])

        actual = y.value
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
