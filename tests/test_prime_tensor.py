import unittest

import numpy as np
import tensorflow as tf
from tensorflow_encrypted.tensor.prime import PrimeTensor


class TestPrimeTensor(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def setUpIndexing(self):
        M = 2 ** 31
        self.np_fix1d = np.arange(24)
        self.np_fix2d = self.np_fix1d.reshape(8, 3)
        self.np_fix3d = self.np_fix1d.reshape(2, 4, 3)
        self.np_fix4d = self.np_fix1d.reshape(2, 2, 2, 3)
        self.prime_fix1d = PrimeTensor.from_native(self.np_fix1d, M)
        self.prime_fix2d = PrimeTensor.from_native(self.np_fix2d, M)
        self.prime_fix3d = PrimeTensor.from_native(self.np_fix3d, M)
        self.prime_fix4d = PrimeTensor.from_native(self.np_fix4d, M)
        self.np_fixtures = [getattr(self, 'np_fix{}d'.format(i)) for i in range(1, 5)]
        self.prime_fixtures = [getattr(self, 'prime_fix{}d'.format(i)) for i in range(1, 5)]

    def test_basic_indexing(self):
        self.setUpIndexing()
        for np_fix, prime_fix in zip(self.np_fixtures, self.prime_fixtures):
            n = len(np_fix.shape)
            for filler in [0, 1, -1]:
                ixs = [filler for _ in range(n)]
                np.testing.assert_equal(np_fix[ixs], prime_fix[ixs])

    def test_slice_indexing(self):
        self.setUpIndexing()
        for np_fix, prime_fix in zip(self.np_fixtures, self.prime_fixtures):
            ndim = len(np_fix.shape)
            if ndim == 1:
                np.testing.assert_equal(np_fix[2:5], prime_fix[2:5])
                continue
            np.testing.assert_equal(np_fix[:, 0], prime_fix[:, 0])
            np.testing.assert_equal(np_fix[:, 1], prime_fix[:, 1])
            np.testing.assert_equal(np_fix[:, -1], prime_fix[:, -1])
            if ndim > 2:
                np.testing.assert_equal(np_fix[:, :-1, ...], prime_fix[:, :-1, ...])
                np.testing.assert_equal(np_fix[:, :1, ...], prime_fix[:, :1, ...])
                np.testing.assert_equal(np_fix[:, 1:, ...], prime_fix[:, 1:, ...])
            elif ndim == 2:
                np.testing.assert_equal(np_fix[:, :2], prime_fix[:, :-1])
                np.testing.assert_equal(np_fix[:, 1:], prime_fix[:, 1:])

    def test_ellipsis_indexing(self):
        self.setUpIndexing()
        for np_fix, prime_fix in zip(self.np_fixtures, self.prime_fixtures):
            np.testing.assert_equal(np_fix[0, ...], prime_fix[0, ...])
            np.testing.assert_equal(np_fix[1, ...], prime_fix[1, ...])
            np.testing.assert_equal(np_fix[..., -1], prime_fix[..., -1])

    def test_arithmetic(self) -> None:
        x = PrimeTensor(tf.constant([2**16, 2**16 + 1]), 2**16)
        y = PrimeTensor(tf.constant([2**16 + 2, 2]), 2**16)

        with tf.Session() as sess:
            z = (x * y).value
            z0 = sess.run(z)

            np.testing.assert_array_equal(z0, np.array([0, 2]))

            z = (x + y).value
            z1 = sess.run(z)

            np.testing.assert_array_equal(z1, np.array([2, 3]))

            z = (x - y).value
            z2 = sess.run(z)

            np.testing.assert_array_equal(z2, np.array([65534, 65535]))


if __name__ == '__main__':
    unittest.main()
