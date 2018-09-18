import unittest

import numpy as np
import tensorflow as tf
from tensorflow_encrypted.tensor.native import NativeTensor


class TestNativeTensor(unittest.TestCase):
    def setUpIndexing(self):
        M = 2 ** 31
        self.np_fix1d = np.arange(24)
        self.np_fix2d = self.np_fix1d.reshape(8, 3)
        self.np_fix3d = self.np_fix1d.reshape(2, 4, 3)
        self.np_fix4d = self.np_fix1d.reshape(2, 2, 2, 3)
        self.native_fix1d = NativeTensor.from_native(self.np_fix1d, M)
        self.native_fix2d = NativeTensor.from_native(self.np_fix2d, M)
        self.native_fix3d = NativeTensor.from_native(self.np_fix3d, M)
        self.native_fix4d = NativeTensor.from_native(self.np_fix4d, M)
        self.np_fixtures = [getattr(self, 'np_fix{}d'.format(i)) for i in range(1, 5)]
        self.native_fixtures = [getattr(self, 'native_fix{}d'.format(i)) for i in range(1, 5)]

    def test_basic_indexing(self):
        self.setUpIndexing()
        for np_fix, native_fix in zip(self.np_fixtures, self.native_fixtures):
            n = len(np_fix.shape)
            for filler in [0, 1, -1]:
                ixs = [filler for _ in range(n)]
                np.testing.assert_equal(np_fix[ixs], native_fix[ixs])

    def test_slice_indexing(self):
        self.setUpIndexing()
        for np_fix, native_fix in zip(self.np_fixtures, self.native_fixtures):
            ndim = len(np_fix.shape)
            if ndim == 1:
                np.testing.assert_equal(np_fix[2:5], native_fix[2:5])
                continue
            np.testing.assert_equal(np_fix[:, 0], native_fix[:, 0])
            np.testing.assert_equal(np_fix[:, 1], native_fix[:, 1])
            np.testing.assert_equal(np_fix[:, -1], native_fix[:, -1])
            if ndim > 2:
                np.testing.assert_equal(np_fix[:, :-1, ...], native_fix[:, :-1, ...])
                np.testing.assert_equal(np_fix[:, :1, ...], native_fix[:, :1, ...])
                np.testing.assert_equal(np_fix[:, 1:, ...], native_fix[:, 1:, ...])
            elif ndim == 2:
                np.testing.assert_equal(np_fix[:, :2], native_fix[:, :-1])
                np.testing.assert_equal(np_fix[:, 1:], native_fix[:, 1:])

    def test_ellipsis_indexing(self):
        self.setUpIndexing()
        for np_fix, native_fix in zip(self.np_fixtures, self.native_fixtures):
            np.testing.assert_equal(np_fix[0, ...], native_fix[0, ...])
            np.testing.assert_equal(np_fix[1, ...], native_fix[1, ...])
            np.testing.assert_equal(np_fix[..., -1], native_fix[..., -1])

    def test_arithmetic(self) -> None:
        x = NativeTensor(tf.constant([2**16, 2**16 + 1]), 2**16)
        y = NativeTensor(tf.constant([2**16 + 2, 2]), 2**16)

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

    def test_binaraize(self) -> None:
        x = NativeTensor(tf.constant([2, 1000]), 2**16)
        y = x.binarize()

        expected = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        with tf.Session() as sess:
            actual = sess.run(y.value)

        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
