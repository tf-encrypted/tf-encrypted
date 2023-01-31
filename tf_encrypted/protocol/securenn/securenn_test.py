# pylint: disable=missing-docstring
import random
import unittest

import numpy as np
import pytest
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.protocol.pond import PondPrivateTensor
from tf_encrypted.protocol.pond import PondPublicTensor
from tf_encrypted.protocol.securenn.securenn import _private_compare


@pytest.mark.securenn
class TestPrivateCompare(unittest.TestCase):
    def test_int64(self):
        self._core_test("low")

    def test_int100(self):
        self._core_test("high")

    def _core_test(self, fixedpoint_config):
        prot = tfe.protocol.SecureNN(fixedpoint_config=fixedpoint_config)
        tfe.set_protocol(prot)

        bit_dtype = prot.prime_factory
        val_dtype = prot.tensor_factory

        x = np.array(
            [21, 21, 21, 21, 21, 21, 21, 21],
            dtype=np.int32,
        ).reshape((2, 2, 2))

        r = np.array(
            [36, 20, 21, 22, 36, 20, 21, 22],
            dtype=np.int32,
        ).reshape((2, 2, 2))

        beta = np.array(
            [0, 0, 0, 0, 1, 1, 1, 1],
            dtype=np.int32,
        ).reshape((2, 2, 2))

        expected = np.bitwise_xor(x > r, beta.astype(bool)).astype(np.int32)
        x_native = tf.convert_to_tensor(x, dtype=val_dtype.native_type)
        x_bits_preshare = val_dtype.tensor(x_native).bits(bit_dtype)
        x_bits = prot._share(x_bits_preshare)  # pylint: disable=protected-access

        r_native = tf.convert_to_tensor(r, dtype=val_dtype.native_type)
        r0 = r1 = val_dtype.tensor(r_native)

        beta_native = tf.convert_to_tensor(beta, dtype=bit_dtype.native_type)
        beta0 = beta1 = bit_dtype.tensor(beta_native)

        res = _private_compare(
            prot,
            x_bits=PondPrivateTensor(prot, *x_bits, False),
            r=PondPublicTensor(prot, r0, r1, False),
            beta=PondPublicTensor(prot, beta0, beta1, False),
        )
        actual = res.reveal().value_on_0.to_native()
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.securenn
class TestSelectShare(unittest.TestCase):
    def test_select_share(self):
        prot = tfe.protocol.SecureNN()
        tfe.set_protocol(prot)

        alice = np.array([1, 1, 1, 1]).astype(np.float32)
        bob = np.array([2, 2, 2, 2]).astype(np.float32)
        bit = np.array([1, 0, 1, 0]).astype(np.float32)
        expected = np.array([2, 1, 2, 1]).astype(np.float32)

        alice_input = prot.define_private_variable(alice, apply_scaling=True)
        bob_input = prot.define_private_variable(bob, apply_scaling=True)
        bit_input = prot.define_private_variable(bit, apply_scaling=False)

        select = prot.select(bit_input, alice_input, bob_input)
        chosen = select.reveal().to_native()
        np.testing.assert_equal(expected, chosen)


@pytest.mark.securenn
class TestLSB(unittest.TestCase):
    def _core_lsb(self, fixedpoint_config, prime_factory):

        f_bin = np.vectorize(np.binary_repr)
        f_get = np.vectorize(lambda x, ix: x[ix])

        raw = np.array([random.randrange(0, 10000000000) for _ in range(20)])
        raw = raw.reshape((2, 2, 5))
        expected_lsb = f_get(f_bin(raw), -1).astype(np.int32)

        with tfe.protocol.SecureNN(
            fixedpoint_config=fixedpoint_config,
            prime_factory=prime_factory,
        ) as prot:

            x_in = prot.define_private_variable(
                raw, apply_scaling=False, name="test_lsb_input"
            )
            x_lsb = prot.lsb(x_in)
            actual_lsb = x_lsb.reveal().to_native()
            np.testing.assert_array_equal(actual_lsb, expected_lsb)

    def test_lsb_int100(self):
        self._core_lsb("high", None)


@pytest.mark.securenn
class TestArgMax(unittest.TestCase):
    @unittest.skipUnless(
        tfe.config.tensorflow_supports_int64(), "Too slow on Circle CI otherwise"
    )
    def test_argmax_1d(self):

        t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(float)
        expected = tf.argmax(t)

        with tfe.protocol.SecureNN() as prot:
            out_tfe = prot.argmax(prot.define_private_variable(tf.constant(t)))
            actual = out_tfe.reveal().to_native()

        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(
        tfe.config.tensorflow_supports_int64(), "Too slow on Circle CI otherwise"
    )
    def test_argmax_2d_axis0(self):

        t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 4).astype(float)
        expected = tf.argmax(t, axis=0)

        with tfe.protocol.SecureNN() as prot:
            out_tfe = prot.argmax(prot.define_private_variable(tf.constant(t)), axis=0)
            actual = out_tfe.reveal().to_native()

        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(
        tfe.config.tensorflow_supports_int64(), "Too slow on Circle CI otherwise"
    )
    def test_argmax_2d_axis1(self):

        t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 4).astype(float)
        expected = tf.argmax(t, axis=1)

        with tfe.protocol.SecureNN() as prot:
            out_tfe = prot.argmax(prot.define_private_variable(tf.constant(t)), axis=1)
            actual = out_tfe.reveal().to_native()

        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(
        tfe.config.tensorflow_supports_int64(), "Too slow on Circle CI otherwise"
    )
    def test_argmax_3d_axis0(self):

        t = np.array(np.arange(128)).reshape((8, 2, 2, 2, 2))
        expected = tf.argmax(t, axis=0)

        with tfe.protocol.SecureNN() as prot:
            out_tfe = prot.argmax(prot.define_private_variable(tf.constant(t)), axis=0)
            actual = out_tfe.reveal().to_native()

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
