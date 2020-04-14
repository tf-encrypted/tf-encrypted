# pylint: disable=missing-docstring
import contextlib
import unittest

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from tf_encrypted.primitives.sodium.python import easy_box


def tf_execution_mode(eager):
    if not eager:
        return tf.Graph().as_default()
    return contextlib.suppress()


class TestExecutionMode(parameterized.TestCase):
    @parameterized.parameters(True, False)
    def test_tf_execution_mode(self, eager: bool):
        with tf_execution_mode(eager):
            assert tf.executing_eagerly() == eager


class TestEasyBox(parameterized.TestCase):
    @parameterized.parameters(
        {"eager": True}, {"eager": False},
    )
    def test_gen_keypair(self, eager):
        with tf_execution_mode(eager):
            pk, sk = easy_box.gen_keypair()

        assert isinstance(pk, easy_box.PublicKey), type(pk)
        assert pk.raw.dtype == tf.uint8
        assert pk.raw.shape == (32,)

        assert isinstance(sk, easy_box.SecretKey), type(sk)
        assert sk.raw.dtype == tf.uint8
        assert sk.raw.shape == (32,)

    @parameterized.parameters(True, False)
    def test_gen_nonce(self, eager):
        with tf_execution_mode(eager):
            nonce = easy_box.gen_nonce()

        assert nonce.raw.dtype == tf.uint8
        assert nonce.raw.shape == (24,)
        assert isinstance(nonce, easy_box.Nonce), type(nonce)
        assert isinstance(nonce.raw, tf.Tensor)

    @parameterized.parameters(
        {"eager": eager, "m": m, "dtype": dtype, "dtype_size": dtype_size}
        for eager in (True, False)
        for m in (5, [5], [[1, 2], [3, 4]])
        for dtype, dtype_size in [(tf.uint8, 1), (tf.float32, 4)]
    )
    def test_seal_and_open(self, eager, m, dtype, dtype_size):
        with tf_execution_mode(eager):
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
        if eager:
            np.testing.assert_equal(plaintext_recovered, np.array(m))


if __name__ == "__main__":
    unittest.main()
