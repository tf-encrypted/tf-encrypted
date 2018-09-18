import unittest

import numpy as np
import tensorflow as tf
from tensorflow_encrypted.tensor.native import NativeTensor


class TestNativeTensor(unittest.TestCase):
    def test_arithmetic(self) -> None:
        tf.enable_eager_execution()

        x = NativeTensor(tf.constant([2**16, 2**16 + 1]), 2**16)
        y = NativeTensor(tf.constant([2**16 + 2, 2]), 2**16)

        z = x * y

        np.testing.assert_array_equal(z.value, np.array([0, 2]))

        z = x + y

        np.testing.assert_array_equal(z.value, np.array([2, 3]))

        z = x - y

        np.testing.assert_array_equal(z.value, np.array([65534, 65535]))

    def test_binaraize(self) -> None:
        tf.enable_eager_execution()

        x = NativeTensor(tf.constant([2, 1000]), 2**16)
        y = x.binarize()

        expected = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        np.testing.assert_array_equal(y.value, expected)


if __name__ == '__main__':
    unittest.main()
