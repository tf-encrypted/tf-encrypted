# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.protocol.pond import PondPublicTensor
from tf_encrypted.tensor import int64factory, int100factory, native_factory
from tf_encrypted.tensor import fixed100, fixed100_ni
from tf_encrypted.utils import unwrap_fetches

def test_encode():
  tf.compat.v1.enable_v2_behavior()
  prot = tfe.protocol.Pond()

  expected = np.array([1234567.9875])
  @tf.function
  def func():
    x = prot.define_constant(expected)

    return unwrap_fetches(x)

  np.testing.assert_array_almost_equal(func(), expected, decimal=3)


class TestTruncate(unittest.TestCase):
  def setUp(self):
    tf.compat.v1.enable_v2_behavior()

  def test_interactive_truncate(self):

    prot = tfe.protocol.Pond(
        tensor_factory=int100factory,
        fixedpoint_config=fixed100,
    )

    # TODO[Morten] remove this condition
    if prot.tensor_factory not in [tfe.tensor.int64factory]:
      expected = np.array([12345.6789])
      @tf.function
      def func():
        w = prot.define_private_tensor(
          expected * prot.fixedpoint_config.scaling_factor)  # double precision
        v = prot.truncate(w)  # single precision

        return unwrap_fetches(v.reveal())

      np.testing.assert_allclose(func(), expected)

  def test_noninteractive_truncate(self):

    prot = tfe.protocol.Pond(
        tensor_factory=int100factory,
        fixedpoint_config=fixed100_ni,
    )

    expected = np.array([12345.6789])

    @tf.function
    def func():
      w = prot.define_private_tensor(
        expected * prot.fixedpoint_config.scaling_factor)  # double precision
      v = prot.truncate(w)  # single precision

      return unwrap_fetches(v.reveal())

    np.testing.assert_allclose(func(), expected)


class TestPondPublicEqual(unittest.TestCase):
  def setUp(self):
    tf.compat.v1.enable_v2_behavior()

  def test_public_compare(self):

    expected = np.array([1, 0, 1, 0])

    with tfe.protocol.Pond() as prot:

      @tf.function
      def func():
        x_raw = prot.tensor_factory.constant(np.array([100, 200, 100, 300]))
        x = PondPublicTensor(prot, value_on_0=x_raw,
                             value_on_1=x_raw, is_scaled=False)

        res = prot.equal(x, 100)

        return unwrap_fetches(res)

      assert np.array_equal(func(), expected)


class TestPondPublicDivision(unittest.TestCase):
  def setUp(self):
    tf.compat.v1.enable_v2_behavior()

  def test_public_division(self):
    x_raw = np.array([10., 20., 30., 40.])
    y_raw = np.array([1., 2., 3., 4.])
    expected = x_raw / y_raw

    with tfe.protocol.Pond() as prot:

      @tf.function
      def func():
        x = prot.define_private_tensor(x_raw)
        y = prot.define_constant(y_raw)
        z = x / y

        return unwrap_fetches(z.reveal())

      np.testing.assert_array_almost_equal(func(), expected, decimal=2)

  def test_public_reciprocal(self):
    x_raw = np.array([10., 20., 30., 40.])
    expected = 1. / x_raw

    with tfe.protocol.Pond() as prot:

      @tf.function
      def func():
        x = prot.define_constant(x_raw)
        y = prot.reciprocal(x)

        return unwrap_fetches(y)

      np.testing.assert_array_almost_equal(func(), expected, decimal=3)

class TestShare(unittest.TestCase):
  def setUp(self):
    tf.compat.v1.enable_v2_behavior()

  def _core_test_sharing(self, dtype):

    expected = np.array([[1, 2, 3], [4, 5, 6]])

    with tfe.protocol.Pond() as prot:

      @tf.function
      def func():
        shares = prot._share(dtype.tensor(expected))  # pylint: disable=protected-access
        return prot._reconstruct(*shares).to_native()  # pylint: disable=protected-access

    np.testing.assert_array_equal(func(), expected)

  def test_int64(self):
    self._core_test_sharing(int64factory)

  def test_int100(self):
    self._core_test_sharing(int100factory)

  def test_prime(self):
    self._core_test_sharing(native_factory(tf.int32, 67))


class TestIdentity(unittest.TestCase):

  def setUp(self):
    tf.compat.v1.reset_default_graph()

  def test_same_value_different_instance(self):

    expected = np.array([[1, 2, 3], [4, 5, 6]])

    with tfe.protocol.Pond() as prot:

      @tf.function
      def func():
        x = prot.define_private_tensor(expected)
        y = prot.identity(x)

        assert x is not y

        return unwrap_fetches(y.reveal())

    np.testing.assert_array_equal(func(), expected)
