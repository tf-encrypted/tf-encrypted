import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.protocol.pond import PondPrivateTensor, PondPublicTensor
from tf_encrypted.protocol.securenn.securenn import _private_compare


class TestPrivateCompare(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def test_int64(self):
        self._core_test(tfe.tensor.int64factory)

    def test_int100(self):
        self._core_test(tfe.tensor.int100factory)

    def _core_test(self, tensor_factory):

        prot = tfe.protocol.SecureNN(tensor_factory=tensor_factory)

        bit_dtype = prot.prime_factory
        val_dtype = prot.tensor_factory

        x = np.array([
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21
        ], dtype=np.int32).reshape(2, 2, 2)

        r = np.array([
            36,
            20,
            21,
            22,
            36,
            20,
            21,
            22
        ], dtype=np.int32).reshape(2, 2, 2)

        beta = np.array([
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1
        ], dtype=np.int32).reshape(2, 2, 2)

        expected = np.bitwise_xor(x > r, beta.astype(bool)).astype(np.int32)

        res = _private_compare(
            prot,
            x_bits=PondPrivateTensor(
                prot,
                *prot._share(val_dtype.tensor(tf.convert_to_tensor(x, dtype=val_dtype.native_type)).bits(bit_dtype)),
                False),
            r=PondPublicTensor(
                prot,
                val_dtype.tensor(tf.convert_to_tensor(r, dtype=val_dtype.native_type)),
                val_dtype.tensor(tf.convert_to_tensor(r, dtype=val_dtype.native_type)),
                False),
            beta=PondPublicTensor(
                prot,
                bit_dtype.tensor(tf.convert_to_tensor(beta, dtype=bit_dtype.native_type)),
                bit_dtype.tensor(tf.convert_to_tensor(beta, dtype=bit_dtype.native_type)),
                False)
        )

        with tfe.Session() as sess:
            actual = sess.run(res.reveal().value_on_0.to_native())
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
