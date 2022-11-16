# pylint: disable=missing-docstring
# pylint: disable=protected-access

import unittest

import numpy as np
import pytest
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.protocol.pond import PondMaskedTensor
from tf_encrypted.protocol.pond import PondPublicTensor
from tf_encrypted.tensor import fixed100
from tf_encrypted.tensor import fixed100_ni
from tf_encrypted.tensor import int64factory
from tf_encrypted.tensor import int100factory
from tf_encrypted.tensor import native_factory


@pytest.mark.pond
class TestPond(unittest.TestCase):
    def test_encode(self):
        prot = tfe.protocol.Pond()
        tfe.set_protocol(prot)

        expected = np.array([1234567.9875])
        x = prot.define_constant(expected)
        actual = x.to_native()
        np.testing.assert_array_almost_equal(actual, expected, decimal=3)


@pytest.mark.pond
class TestTruncate(unittest.TestCase):
    def test_interactive_truncate(self):
        prot = tfe.protocol.Pond(
            tensor_factory=int100factory, fixedpoint_config=fixed100
        )
        tfe.set_protocol(prot)

        # TODO[Morten] remove this condition
        if prot.tensor_factory not in [tfe.tensor.int64factory]:

            expected = np.array([12345.6789])

            w = prot.define_private_variable(
                expected * prot.fixedpoint_config.scaling_factor
            )  # double precision
            v = prot.truncate(w)  # single precision
            actual = v.reveal().to_native()
            np.testing.assert_allclose(actual, expected)

    def test_noninteractive_truncate(self):
        prot = tfe.protocol.Pond(
            tensor_factory=int100factory, fixedpoint_config=fixed100_ni
        )
        tfe.set_protocol(prot)

        expected = np.array([12345.6789])
        w = prot.define_private_variable(
            expected * prot.fixedpoint_config.scaling_factor
        )  # double precision
        v = prot.truncate(w)  # single precision
        actual = v.reveal().to_native()
        np.testing.assert_allclose(actual, expected)


@pytest.mark.pond
class TestPondPublicEqual(unittest.TestCase):
    def test_public_compare(self):
        prot = tfe.protocol.Pond()
        tfe.set_protocol(prot)

        expected = np.array([1, 0, 1, 0])
        x_raw = prot.tensor_factory.constant(np.array([100, 200, 100, 300]))
        x = PondPublicTensor(prot, value_on_0=x_raw, value_on_1=x_raw, is_scaled=False)
        res = prot.equal(x, 100)
        answer = res.to_native()
        assert np.array_equal(answer, expected)


@pytest.mark.pond
class TestPondPublicDivision(unittest.TestCase):
    def test_public_division(self):
        prot = tfe.protocol.Pond()
        tfe.set_protocol(prot)

        x_raw = np.array([10.0, 20.0, 30.0, 40.0])
        y_raw = np.array([1.0, 2.0, 3.0, 4.0])
        expected = x_raw / y_raw
        x = prot.define_private_variable(x_raw)
        y = prot.define_constant(y_raw)
        z = x / y
        actual = z.reveal().to_native()
        np.testing.assert_array_almost_equal(actual, expected, decimal=2)

    def test_public_reciprocal(self):
        prot = tfe.protocol.Pond()
        tfe.set_protocol(prot)

        x_raw = np.array([10.0, 20.0, 30.0, 40.0])
        expected = 1.0 / x_raw
        x = prot.define_constant(x_raw)
        y = prot.reciprocal(x)
        actual = y.to_native()
        np.testing.assert_array_almost_equal(actual, expected, decimal=3)


@pytest.mark.pond
class TestShare(unittest.TestCase):
    def _core_test_sharing(self, dtype):
        prot = tfe.protocol.Pond()
        tfe.set_protocol(prot)

        expected = np.array([[1, 2, 3], [4, 5, 6]])
        shares = prot._share(dtype.tensor(expected))
        actual = prot._reconstruct(*shares).to_native()
        np.testing.assert_array_equal(actual, expected)

    def test_int64(self):
        self._core_test_sharing(int64factory)

    def test_int100(self):
        self._core_test_sharing(int100factory)

    def test_prime(self):
        self._core_test_sharing(native_factory(tf.int32, 67))


@pytest.mark.pond
class TestMasked(unittest.TestCase):
    def _setup(self, dtype):
        prot = tfe.protocol.Pond()
        tfe.set_protocol(prot)

        plain_tensor = dtype.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        # pylint: disable=protected-access
        unmasked = prot._share_and_wrap(plain_tensor, False)
        # pylint: enable=protected-access
        a0 = dtype.sample_uniform(plain_tensor.shape)
        a1 = dtype.sample_uniform(plain_tensor.shape)
        a = a0 + a1
        alpha = plain_tensor - a
        x = PondMaskedTensor(prot, unmasked, a, a0, a1, alpha, alpha, False)
        return prot, x

    def test_transpose_masked(self):

        prot, x = self._setup(int64factory)
        transpose = prot.transpose(x)
        actual = transpose.reveal().to_native()
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(actual, expected)

    def test_indexer(self):

        prot, x = self._setup(int64factory)
        indexed = tfe.indexer(x, 0)
        actual = indexed.reveal().to_native()
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(actual, expected)

    def test_gather(self):

        prot, x = self._setup(int64factory)
        gathered = tfe.gather(x, [0, 2], axis=1)
        actual = gathered.reveal().to_native()
        expected = np.array([[1, 3], [4, 6]])
        np.testing.assert_array_equal(actual, expected)

    def test_negative_masked(self):

        prot, x = self._setup(int64factory)
        negative = tfe.negative(x)
        actual = negative.reveal().to_native()
        expected = np.array([[-1, -2, -3], [-4, -5, -6]])
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.pond
class TestIdentity(unittest.TestCase):
    def test_same_value_different_instance(self):
        prot = tfe.protocol.Pond()
        tfe.set_protocol(prot)

        expected = np.array([[1, 2, 3], [4, 5, 6]])
        x = prot.define_private_variable(expected)
        y = prot.identity(x)
        actual = y.reveal().to_native()
        assert x is not y
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.pond
class TestPondAssign(unittest.TestCase):
    def test_assign_synchronization(self):
        # from https://github.com/tf-encrypted/tf-encrypted/pull/665

        prot = tfe.protocol.Pond()
        tfe.set_protocol(prot)

        def poc(x, y):
            x_shares = x.unwrapped
            y_shares = y.unwrapped
            z_shares = [None, None]

            with tf.name_scope("fabricated_test"):
                with tf.device(prot.server_0.device_name):
                    z_shares[0] = x_shares[1] + y_shares[1]
                with tf.device(prot.server_1.device_name):
                    z_shares[1] = x_shares[0] + y_shares[0]

            return tfe.protocol.pond.PondPrivateTensor(
                prot, z_shares[0], z_shares[1], x.is_scaled
            )

        a = prot.define_private_variable(tf.ones(shape=(1, 1)))
        b = prot.define_private_variable(tf.ones(shape=(1, 1)))
        for _ in range(100):
            prot.assign(a, poc(a.read_value(), b.read_value()))
        result = a.read_value().reveal().to_native()
        assert result == np.array([101.0])

    def test_public_assign(self):
        prot = tfe.protocol.Pond()
        tfe.set_protocol(prot)
        x_var = prot.define_public_variable(np.zeros(shape=(2, 2)))
        data = np.ones((2, 2))
        x = tfe.define_constant(data)
        tfe.assign(x_var, x)
        result = x_var.read_value().to_native()
        np.testing.assert_array_equal(result, np.ones([2, 2]))


if __name__ == "__main__":
    tfe.get_config().set_debug_mode(True)
    unittest.main()
