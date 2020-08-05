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
        }
        for run_eagerly in [True, False]
        for export_dtype, export_expansion in [(tf.string, ())]
    )
    def test_export(self, run_eagerly, export_dtype, export_expansion):
        x = np.array([[12345, 34342]])

        context = tf_execution_context(run_eagerly)
        with context.scope():

            ek, dk = paillier.gen_keypair()
            assert isinstance(ek, paillier.EncryptionKey)
            assert isinstance(dk, paillier.DecryptionKey)
            n_exported = ek.export(export_dtype)
            assert isinstance(n_exported, tf.Tensor)
            assert n_exported.dtype == export_dtype
            assert n_exported.shape == (1, 1), n_exported.shape
            p_exported, q_exported = dk.export(export_dtype)
            assert isinstance(p_exported, tf.Tensor)
            assert p_exported.dtype == export_dtype
            assert p_exported.shape == (1, 1), p_exported.shape
            assert isinstance(q_exported, tf.Tensor)
            assert q_exported.dtype == export_dtype
            assert q_exported.shape == (1, 1), q_exported.shape

            r = paillier.gen_randomness(ek, shape=x.shape)
            assert isinstance(r, paillier.Randomness)
            r_exported = r.export(export_dtype)
            assert isinstance(r_exported, tf.Tensor)
            assert r_exported.dtype == export_dtype
            assert r_exported.shape == x.shape + export_expansion

            c = paillier.encrypt(ek, x, r)
            assert isinstance(c, paillier.Ciphertext)
            c_exported = c.export(export_dtype)
            assert isinstance(c_exported, tf.Tensor)
            assert c_exported.dtype == export_dtype
            assert c_exported.shape == x.shape + export_expansion

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

            ek = paillier.EncryptionKey(tf.constant([[str(n)]]))
            plaintext = np.array([[x]]).astype(str)
            randomness = paillier.Randomness(tf.constant([[str(r)]]))
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
