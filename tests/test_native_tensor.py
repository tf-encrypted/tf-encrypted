import unittest

import numpy as np
import tensorflow as tf
from tensorflow_encrypted.tensor.native import NativeTensor


class TestNativeTensor(unittest.TestCase):
    def setUp(self):
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
        for np_fix, native_fix in zip(self.np_fixtures, self.native_fixtures):
            n = len(np_fix.shape)
            for filler in [0, 1, -1]:
                ixs = [filler for _ in range(n)]
                np.testing.assert_equal(np_fix[ixs], native_fix[ixs])

    def test_colon_indexing(self):
        for np_fix, native_fix in zip(self.np_fixtures, self.native_fixtures):
            if len(np_fix.shape) == 1:
                continue
            np.testing.assert_equal(np_fix[:, 0], native_fix[:, 0])
            np.testing.assert_equal(np_fix[:, 1], native_fix[:, 1])
            np.testing.assert_equal(np_fix[:, -1], native_fix[:, -1])


    def test_ellipsis_indexing(self):
        for np_fix, native_fix in zip(self.np_fixtures, self.native_fixtures):
            np.testing.assert_equal(np_fix[0, ...], native_fix[0, ...])
            np.testing.assert_equal(np_fix[1, ...], native_fix[1, ...])
            np.testing.assert_equal(np_fix[..., -1], native_fix[..., -1])

if __name__ == '__main__':
    unittest.main()
