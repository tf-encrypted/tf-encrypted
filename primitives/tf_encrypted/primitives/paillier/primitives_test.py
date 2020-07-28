# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from tf_encrypted.primitives import paillier
from tf_encrypted.test import tf_execution_context


class EncryptionTest(parameterized.TestCase):
    @parameterized.parameters(
        {
            "run_eagerly": run_eagerly,
            "export_dtype": export_dtype,
            "export_expansion": export_expansion,
            "c_expansion": c_expansion,
            "bitlength": bitlength,
        }
        for run_eagerly in [True, False]
        for export_dtype, export_expansion, c_expansion, bitlength in [
            (tf.string, (), (), None),
            (tf.int32, (3,), (5,), 64),
            (tf.uint8, (12,), (20,), 64)
        ]
    )
    def test_export(self, run_eagerly, export_dtype, export_expansion, c_expansion, bitlength):
        x = np.array([[12345, 34342]])

        context = tf_execution_context(run_eagerly)
        with context.scope():

            ek, dk = paillier.gen_keypair(bitlength or 64)
            assert isinstance(ek, paillier.EncryptionKey)
            assert isinstance(dk, paillier.DecryptionKey)
            n_exported = ek.export(export_dtype, bitlength=bitlength)
            assert isinstance(n_exported, tf.Tensor)
            assert n_exported.dtype == export_dtype
            n_realized = context.evaluate(n_exported)
            assert n_realized.shape == (1, 1) + export_expansion
            p_exported, q_exported = dk.export(export_dtype, bitlength=bitlength)
            assert isinstance(p_exported, tf.Tensor)
            assert p_exported.dtype == export_dtype
            p_realized = context.evaluate(p_exported)
            assert p_realized.shape == (1, 1) + export_expansion
            assert isinstance(q_exported, tf.Tensor)
            assert q_exported.dtype == export_dtype
            q_realized = context.evaluate(q_exported)
            assert q_realized.shape == (1, 1) + export_expansion

            r = paillier.gen_randomness(ek, shape=x.shape)
            assert isinstance(r, paillier.Randomness)
            r_exported = r.export(export_dtype, bitlength=bitlength)
            assert isinstance(r_exported, tf.Tensor)
            assert r_exported.dtype == export_dtype
            r_realized = context.evaluate(r_exported)
            assert r_realized.shape == x.shape + export_expansion

            bl = 2 * bitlength if bitlength is not None else None
            c = paillier.encrypt(ek, x, r)
            assert isinstance(c, paillier.Ciphertext)
            c_exported = c.export(export_dtype, bitlength=bl)
            assert isinstance(c_exported, tf.Tensor)
            assert c_exported.dtype == export_dtype
            c_realized = context.evaluate(c_exported)
            assert c_realized.shape == x.shape + c_expansion

    @parameterized.parameters(
        {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
    )
    def test_correctness(self, run_eagerly):

        p = 100000015333
        q = 100000015021
        n = p * q
        nn = n * n
        g = 1 + n
        x = 123456789
        r = 5083216764521909821749
        c = pow(g, x, nn) * pow(r, n, nn) % nn

        context = tf_execution_context(run_eagerly)
        with context.scope():

            ek = paillier.EncryptionKey(str(n))
            plaintext = np.array([[x]]).astype(str)
            randomness = paillier.Randomness(np.array([[r]]).astype(str))
            ciphertext = paillier.encrypt(ek, plaintext, randomness)

            expected = np.array([[c]]).astype(str)
            actual = ciphertext.export(tf.string)

        np.testing.assert_equal(context.evaluate(actual).astype(str), expected)

    @parameterized.parameters(
        {"run_eagerly": run_eagerly, "x": x, "dtype": dtype}
        for run_eagerly in [True, False]
        for x, dtype in [
            (np.array([[12345, 34342]]).astype(np.int32), tf.int32),
            (np.array([["12345", "34342"]]).astype(str), tf.string),
            (
                np.array(
                    [
                        [
                            "123456789123456789123456789123456789",
                            "987654321987654321987654321987654321",
                        ]
                    ]
                ).astype(str),
                tf.string,
            ),
        ]
    )
    def test_encrypt_decrypt(self, run_eagerly, x, dtype):
        context = tf_execution_context(run_eagerly)
        with context.scope():

            ek, dk = paillier.gen_keypair()
            r = paillier.gen_randomness(ek, shape=x.shape)
            c = paillier.encrypt(ek, x, r)
            y = paillier.decrypt(dk, c, dtype=dtype)
            assert isinstance(y, tf.Tensor)
            assert y.dtype == dtype

        np.testing.assert_equal(context.evaluate(y).astype(x.dtype), x)

    @parameterized.parameters(
        {"run_eagerly": run_eagerly, "dtype": dtype, "x0": x0, "x1": x1}
        for run_eagerly in (True, False)
        for dtype in (tf.int32, tf.string)
        for x0 in (np.array([[12345, 123243]]), np.array([[12345]]))
        for x1 in (np.array([[12656, 434234]]),)
    )
    def test_add(self, run_eagerly, dtype, x0, x1):

        expected = x0 + x1

        context = tf_execution_context(run_eagerly)
        with context.scope():
            ek, dk = paillier.gen_keypair()

            r0 = paillier.gen_randomness(ek, shape=x0.shape)
            c0 = paillier.encrypt(ek, x0, r0)

            r1 = paillier.gen_randomness(ek, shape=x1.shape)
            c1 = paillier.encrypt(ek, x1, r1)

            c = paillier.add(ek, c0, c1)
            y = paillier.decrypt(dk, c, dtype=dtype)

        np.testing.assert_equal(
            context.evaluate(y).astype(np.int32), expected.astype(np.int32)
        )


if __name__ == "__main__":
    unittest.main()
