# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from tf_encrypted.primitives import paillier
from tf_encrypted.test import tf_execution_context


class EncryptionTest(parameterized.TestCase):
    @parameterized.parameters(
        {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
    )
    def test_encrypt_decrypt(self, run_eagerly):

        x = np.array([[12345]]).astype(np.int32)

        context = tf_execution_context(run_eagerly)
        with context.scope():
            ek, dk = paillier.gen_keypair()

            r = paillier.gen_randomness(ek, shape=x.shape)
            c = paillier.encrypt(ek, x, r)
            y = paillier.decrypt(dk, c, dtype=tf.int32)

        np.testing.assert_equal(context.evaluate(y).astype(np.int32), x)

    @parameterized.parameters(
        {"run_eagerly": run_eagerly} for run_eagerly in (True, False)
    )
    def test_add(self, run_eagerly):

        x0 = np.array([[12345]])
        x1 = np.array([[12345]])
        expected = x0 + x1

        context = tf_execution_context(run_eagerly)
        with context.scope():
            ek, dk = paillier.gen_keypair()

            r0 = paillier.gen_randomness(ek, shape=x0.shape)
            c0 = paillier.encrypt(ek, x0, r0)

            r1 = paillier.gen_randomness(ek, shape=x1.shape)
            c1 = paillier.encrypt(ek, x1, r1)

            c = paillier.add(ek, c0, c1)
            y = paillier.decrypt(dk, c, dtype=tf.int32)

        np.testing.assert_equal(
            context.evaluate(y).astype(np.int32), expected.astype(np.int32)
        )


if __name__ == "__main__":
    unittest.main()
