import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.tensor import int64factory, int100factory, native_factory


class TestShare(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def _core_test_sharing(self, dtype):

        expected = np.array([[1, 2, 3], [4, 5, 6]])

        with tfe.protocol.Pond() as prot:

            with tfe.Session() as sess:
                shares = prot._share(dtype.tensor(expected))
                actual = sess.run(prot._reconstruct(*shares).to_native())

        np.testing.assert_array_equal(actual, expected)

    def test_int64(self):
        self._core_test_sharing(int64factory)

    def test_int100(self):
        self._core_test_sharing(int100factory)

    def test_prime(self):
        self._core_test_sharing(native_factory(tf.int32, 67))


if __name__ == '__main__':
    unittest.main()
