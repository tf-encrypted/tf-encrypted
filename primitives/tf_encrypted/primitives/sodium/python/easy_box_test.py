# pylint: disable=missing-docstring
import unittest
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tf_encrypted.primitives.sodium.python import easy_box


class TestEasyBox(parameterized.TestCase):

    @parameterized.parameters('eager', 'graph')
    def test_gen_keypair(self, mode):
        pk, sk = run_op(easy_box.gen_keypair, mode)

        assert isinstance(pk, easy_box.PublicKey), type(pk)
        assert pk.raw.dtype == tf.uint8
        assert pk.raw.shape == (32,)

        assert isinstance(sk, easy_box.SecretKey), type(sk)
        assert sk.raw.dtype == tf.uint8
        assert sk.raw.shape == (32,)

    @parameterized.parameters('eager', 'graph')
    def test_gen_nonce(self, mode):
        nonce = run_op(easy_box.gen_nonce, mode)

        assert nonce.raw.dtype == tf.uint8
        assert nonce.raw.shape == (24,)
        assert isinstance(nonce, easy_box.Nonce), type(nonce)
        assert isinstance(nonce.raw, tf.Tensor)

    def test_gen_seal_open_graph(self):

        with tf.Graph().as_default():
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

    @parameterized.named_parameters(
        ('float', tf.float32, (2, 2, 4)),
        ('uint8', tf.uint8, (2, 2, 1)),
    )
    def test_seal(self, dtype, expected_shape):
        _, sk_s = easy_box.gen_keypair()
        pk_r, _ = easy_box.gen_keypair()

        plaintext = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=dtype)

        nonce = easy_box.gen_nonce()
        ciphertext, _ = easy_box.seal_detached(plaintext, nonce, pk_r, sk_s)

        assert ciphertext.raw.shape == expected_shape

    @parameterized.named_parameters(
        ('float', tf.float32),
        ('uint8', tf.uint8),
    )
    def test_open(self, dtype):
        pk_s, sk_s = easy_box.gen_keypair()
        pk_r, sk_r = easy_box.gen_keypair()

        plaintext = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=dtype)

        nonce = easy_box.gen_nonce()
        ciphertext, mac = easy_box.seal_detached(plaintext, nonce, pk_r, sk_s)
        plaintext_recovered = easy_box.open_detached(
            ciphertext, mac, nonce, pk_s, sk_r, plaintext.dtype
        )

        assert plaintext_recovered.shape == plaintext.shape
        np.testing.assert_equal(plaintext_recovered, np.array([[1, 2], [3, 4]]))


def run_op(op, mode):
    if mode == 'graph':
        with tf.Graph().as_default():
            res = op()
    else:
        res = op()

    return res


if __name__ == "__main__":
    unittest.main()
