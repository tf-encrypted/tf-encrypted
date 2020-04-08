# pylint: disable=missing-docstring
import unittest
from absl.testing import parameterized
import contextlib

import numpy as np
import tensorflow as tf

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
        {"eager": True},
        {"eager": False},
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
        {"eager": True},
        {"eager": False},
    )
    def test_gen_seal_open_graph(self, eager):
        with tf_execution_mode(eager):
            pk_s, sk_s = easy_box.gen_keypair()
            pk_r, sk_r = easy_box.gen_keypair()

            plaintext = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=tf.float32)

            nonce = easy_box.gen_nonce()
            ciphertext, mac = easy_box.seal_detached(plaintext, nonce, pk_r, sk_s)
            plaintext_recovered = easy_box.open_detached(
                ciphertext, mac, nonce, pk_s, sk_r, plaintext.dtype
            )

        assert ciphertext.raw.shape == plaintext.shape + (4,)
        assert plaintext_recovered.shape == plaintext.shape

    @parameterized.parameters(
        {"eager": True, "dtype": tf.float32, "expected_shape": (2, 2, 4)},
        {"eager": True, "dtype": tf.uint8, "expected_shape": (2, 2, 1)},
        {"eager": False, "dtype": tf.float32, "expected_shape": (2, 2, 4)},
        {"eager": False, "dtype": tf.uint8, "expected_shape": (2, 2, 1)},
    )
    def test_seal(self, eager, dtype, expected_shape):
        with tf_execution_mode(eager):
            _, sk_s = easy_box.gen_keypair()
            pk_r, _ = easy_box.gen_keypair()

            plaintext = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=dtype)

            nonce = easy_box.gen_nonce()
            ciphertext, _ = easy_box.seal_detached(plaintext, nonce, pk_r, sk_s)

        assert ciphertext.raw.shape == expected_shape

    @parameterized.parameters(
        {"eager": True, "dtype": tf.float32},
        {"eager": True, "dtype": tf.uint8},
        {"eager": False, "dtype": tf.float32},
        {"eager": False, "dtype": tf.uint8},
    )
    def test_open(self, eager, dtype):
        with tf_execution_mode(eager):
            pk_s, sk_s = easy_box.gen_keypair()
            pk_r, sk_r = easy_box.gen_keypair()

            plaintext = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=dtype)

            nonce = easy_box.gen_nonce()
            ciphertext, mac = easy_box.seal_detached(plaintext, nonce, pk_r, sk_s)
            plaintext_recovered = easy_box.open_detached(
                ciphertext, mac, nonce, pk_s, sk_r, plaintext.dtype
            )

        assert plaintext_recovered.shape == plaintext.shape
        if eager:
            np.testing.assert_equal(plaintext_recovered, np.array([[1, 2], [3, 4]]))


if __name__ == "__main__":
    unittest.main()
