# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.primitives.sodium.python import easy_box

class TestEasyBox(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_gen_keypair(self):
        pk, sk = easy_box.gen_keypair()
        assert isinstance(pk, easy_box.PublicKey), type(pk)
        assert isinstance(sk, easy_box.SecretKey), type(sk)

        with tf.Session() as sess:
            _ = sess.run([pk.raw, sk.raw])

    def test_gen_nonce(self):
        nonce = easy_box.gen_nonce()
        assert isinstance(nonce, easy_box.Nonce), type(nonce)
        assert isinstance(nonce.raw, tf.Tensor)

        with tf.Session() as sess:
            _ = sess.run(nonce.raw)

    def test_seal(self):
        _, sk_s = easy_box.gen_keypair()
        pk_r, _ = easy_box.gen_keypair()

        plaintext = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=tf.uint8)

        nonce = easy_box.gen_nonce()
        ciphertext, mac = easy_box.seal_detached(plaintext, nonce, pk_r, sk_s)

        with tf.Session() as sess:
            c, _ = sess.run([ciphertext.raw, mac.raw])

            assert c.shape == plaintext.shape

    def test_open(self):
        pk_s, sk_s = easy_box.gen_keypair()
        pk_r, sk_r = easy_box.gen_keypair()

        plaintext = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=tf.uint8)

        nonce = easy_box.gen_nonce()
        ciphertext, mac = easy_box.seal_detached(plaintext, nonce, pk_r, sk_s)
        plaintext_recovered = easy_box.open_detached(ciphertext, mac, nonce, pk_s, sk_r)

        with tf.Session() as sess:
            p = sess.run(plaintext_recovered)

            assert p.shape == plaintext.shape
            np.testing.assert_equal(p, [[1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()
