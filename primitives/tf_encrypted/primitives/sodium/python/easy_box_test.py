# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from tf_encrypted.primitives.sodium.python import easy_box
from tf_encrypted.test import tf_execution_context


class TestEasyBox(parameterized.TestCase):
    @parameterized.parameters(
        {"run_eagerly": True}, {"run_eagerly": False},
    )
    def test_gen_keypair(self, run_eagerly):
        context = tf_execution_context(run_eagerly)
        with context.scope():
            pk, sk = easy_box.gen_keypair()

        assert isinstance(pk, easy_box.PublicKey), type(pk)
        assert isinstance(pk.raw, tf.Tensor)
        assert pk.raw.dtype == tf.uint8
        assert pk.raw.shape == (32,)

        assert isinstance(sk, easy_box.SecretKey), type(sk)
        assert isinstance(sk.raw, tf.Tensor)
        assert sk.raw.dtype == tf.uint8
        assert sk.raw.shape == (32,)

    @parameterized.parameters(
        {"run_eagerly": True}, {"run_eagerly": False},
    )
    def test_gen_nonce(self, run_eagerly):
        context = tf_execution_context(run_eagerly)
        with context.scope():
            nonce = easy_box.gen_nonce()

        assert isinstance(nonce, easy_box.Nonce), type(nonce)
        assert isinstance(nonce.raw, tf.Tensor)
        assert nonce.raw.dtype == tf.uint8
        assert nonce.raw.shape == (24,)

    @parameterized.parameters(
        {"run_eagerly": run_eagerly, "m": m, "dtype": dtype, "dtype_size": dtype_size}
        for run_eagerly in (True, False)
        for m in (5, [5], [[1, 2], [3, 4]])
        for dtype, dtype_size in [
            (tf.uint8, 1),
            (tf.uint16, 2),
            (tf.uint32, 4),
            (tf.uint64, 8),
            (tf.int8, 1),
            (tf.int16, 2),
            (tf.int32, 4),
            (tf.int64, 8),
            (tf.bfloat16, 2),
            (tf.float32, 4),
            (tf.float64, 8),
        ]
    )
    def test_seal_and_open(self, run_eagerly, m, dtype, dtype_size):
        context = tf_execution_context(run_eagerly)
        with context.scope():
            pk_s, sk_s = easy_box.gen_keypair()
            pk_r, sk_r = easy_box.gen_keypair()

            plaintext = tf.constant(m, dtype=dtype)

            nonce = easy_box.gen_nonce()
            ciphertext, mac = easy_box.seal_detached(plaintext, nonce, pk_r, sk_s)

            plaintext_recovered = easy_box.open_detached(
                ciphertext, mac, nonce, pk_s, sk_r, plaintext.dtype
            )

        assert isinstance(ciphertext, easy_box.Ciphertext)
        assert isinstance(ciphertext.raw, tf.Tensor)
        assert ciphertext.raw.dtype == tf.uint8
        assert ciphertext.raw.shape == plaintext.shape + (dtype_size,)

        assert isinstance(mac, easy_box.Mac)
        assert isinstance(mac.raw, tf.Tensor)
        assert mac.raw.dtype == tf.uint8
        assert mac.raw.shape == (16,)

        assert isinstance(plaintext_recovered, tf.Tensor)
        assert plaintext_recovered.dtype == plaintext.dtype
        assert plaintext_recovered.shape == plaintext.shape

        plaintext_recovered = context.evaluate(plaintext_recovered)
        np.testing.assert_equal(plaintext_recovered, np.array(m))


if __name__ == "__main__":
    unittest.main()
