# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.primitives.sodium.python import easy_box


class TestEasyBox(unittest.TestCase):
    def test_gen_keypair_eager(self):
        pk, sk = easy_box.gen_keypair()

        assert isinstance(pk, easy_box.PublicKey), type(pk)
        assert pk.raw.dtype == tf.uint8
        assert pk.raw.shape == (32,)

        assert isinstance(sk, easy_box.SecretKey), type(sk)
        assert sk.raw.dtype == tf.uint8
        assert sk.raw.shape == (32,)

    def test_gen_keypair_graph(self):
        with tf.Graph().as_default():
            pk, sk = easy_box.gen_keypair()

        assert isinstance(pk, easy_box.PublicKey), type(pk)
        assert pk.raw.dtype == tf.uint8
        assert pk.raw.shape == (32,)

        assert isinstance(sk, easy_box.SecretKey), type(sk)
        assert sk.raw.dtype == tf.uint8
        assert sk.raw.shape == (32,)

    def test_gen_nonce_eager(self):
        nonce = easy_box.gen_nonce()

        assert nonce.raw.dtype == tf.uint8
        assert nonce.raw.shape == (24,)
        assert isinstance(nonce, easy_box.Nonce), type(nonce)
        assert isinstance(nonce.raw, tf.Tensor)

    def test_gen_nonce_graph(self):
        with tf.Graph().as_default():
            nonce = easy_box.gen_nonce()

        assert nonce.raw.dtype == tf.uint8
        assert nonce.raw.shape == (24,)
        assert isinstance(nonce, easy_box.Nonce), type(nonce)
        assert isinstance(nonce.raw, tf.Tensor)

    def test_seal_uint8(self):
        _, sk_s = easy_box.gen_keypair()
        pk_r, _ = easy_box.gen_keypair()

        plaintext = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=tf.uint8)

        nonce = easy_box.gen_nonce()
        ciphertext, _ = easy_box.seal_detached(plaintext, nonce, pk_r, sk_s)

        assert ciphertext.raw.shape == plaintext.shape + (1,)

    def test_seal_floats(self):
        _, sk_s = easy_box.gen_keypair()
        pk_r, _ = easy_box.gen_keypair()

        plaintext = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=tf.float32)

        nonce = easy_box.gen_nonce()
        ciphertext, _ = easy_box.seal_detached(plaintext, nonce, pk_r, sk_s)

        assert ciphertext.raw.shape == plaintext.shape + (4,)

    def test_open_uint8(self):
        pk_s, sk_s = easy_box.gen_keypair()
        pk_r, sk_r = easy_box.gen_keypair()

        plaintext = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=tf.uint8)

        nonce = easy_box.gen_nonce()
        ciphertext, mac = easy_box.seal_detached(plaintext, nonce, pk_r, sk_s)
        plaintext_recovered = easy_box.open_detached(
            ciphertext, mac, nonce, pk_s, sk_r, plaintext.dtype
        )

        assert plaintext_recovered.shape == plaintext.shape
        np.testing.assert_equal(plaintext_recovered, np.array([[1, 2], [3, 4]]))

    def test_open_float(self):
        pk_s, sk_s = easy_box.gen_keypair()
        pk_r, sk_r = easy_box.gen_keypair()

        plaintext = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=tf.float32)

        nonce = easy_box.gen_nonce()
        ciphertext, mac = easy_box.seal_detached(plaintext, nonce, pk_r, sk_s)
        plaintext_recovered = easy_box.open_detached(
            ciphertext, mac, nonce, pk_s, sk_r, plaintext.dtype
        )

        assert plaintext_recovered.shape == plaintext.shape
        np.testing.assert_equal(plaintext_recovered, np.array([[1, 2], [3, 4]]))


if __name__ == "__main__":
    unittest.main()
