# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.protocol.pond import PondPublicTensor
from tf_encrypted.tensor import int64factory, int100factory, native_factory
from tf_encrypted.tensor import fixed100, fixed100_ni


class TestTruncate(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_interactive_truncate(self):

    prot = tfe.protocol.Pond(
        tensor_factory=int100factory,
        fixedpoint_config=fixed100,
    )

    # TODO[Morten] remove this condition
    if prot.tensor_factory not in [tfe.tensor.int64factory]:

      expected = np.array([12345.6789])

      w = prot.define_private_variable(
          expected * prot.fixedpoint_config.scaling_factor)  # double precision
      v = prot.truncate(w)  # single precision

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        actual = sess.run(v.reveal())

      np.testing.assert_allclose(actual, expected)

  def test_noninteractive_truncate(self):

    prot = tfe.protocol.Pond(
        tensor_factory=int100factory,
        fixedpoint_config=fixed100_ni,
    )

    with tfe.Session() as sess:

      expected = np.array([12345.6789])

      w = prot.define_private_variable(
          expected * prot.fixedpoint_config.scaling_factor)  # double precision
      v = prot.truncate(w)  # single precision

      sess.run(tf.global_variables_initializer())
      actual = sess.run(v.reveal())

      np.testing.assert_allclose(actual, expected)



class TestPondPublicEqual(unittest.TestCase):

  def test_public_compare(self):

    expected = np.array([1, 0, 1, 0])

    with tfe.protocol.Pond() as prot:

      x_raw = prot.tensor_factory.constant(np.array([100, 200, 100, 300]))
      x = PondPublicTensor(prot, value_on_0=x_raw,
                           value_on_1=x_raw, is_scaled=False)

      res = prot.equal(x, 100)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        answer = sess.run(res)

      assert np.array_equal(answer, expected)


class TestShare(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def _core_test_sharing(self, dtype):

    expected = np.array([[1, 2, 3], [4, 5, 6]])

    with tfe.protocol.Pond() as prot:

      with tfe.Session() as sess:
        shares = prot._share(dtype.tensor(expected))  # pylint: disable=protected-access
        actual = sess.run(prot._reconstruct(*shares).to_native())  # pylint: disable=protected-access

    np.testing.assert_array_equal(actual, expected)

  def test_int64(self):
    self._core_test_sharing(int64factory)

  def test_int100(self):
    self._core_test_sharing(int100factory)

  def test_prime(self):
    self._core_test_sharing(native_factory(tf.int32, 67))



if __name__ == '__main__':
  unittest.main()
