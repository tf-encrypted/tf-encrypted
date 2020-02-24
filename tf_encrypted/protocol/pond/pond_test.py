# pylint: disable=missing-docstring
# pylint: disable=protected-access

import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.protocol.pond import PondPublicTensor, PondMaskedTensor
from tf_encrypted.tensor import int64factory, int100factory, native_factory
from tf_encrypted.tensor import fixed100, fixed100_ni
from .pond import _gather_masked, _indexer_masked, _negative_masked


class TestPond(unittest.TestCase):

  def test_encode(self):

    with tf.Graph().as_default():
      prot = tfe.protocol.Pond()

      expected = np.array([1234567.9875])
      x = prot.define_constant(expected)

      with tfe.Session() as sess:
        actual = sess.run(x)
        np.testing.assert_array_almost_equal(actual, expected, decimal=3)


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
      x = PondPublicTensor(prot,
                           value_on_0=x_raw,
                           value_on_1=x_raw,
                           is_scaled=False)

      res = prot.equal(x, 100)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        answer = sess.run(res)

      assert np.array_equal(answer, expected)


class TestPondPublicDivision(unittest.TestCase):

  def test_public_division(self):

    x_raw = np.array([10., 20., 30., 40.])
    y_raw = np.array([1., 2., 3., 4.])
    expected = x_raw / y_raw

    with tfe.protocol.Pond() as prot:

      x = prot.define_private_variable(x_raw)
      y = prot.define_constant(y_raw)
      z = x / y

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        actual = sess.run(z.reveal())

      np.testing.assert_array_almost_equal(actual, expected, decimal=2)

  def test_public_reciprocal(self):

    x_raw = np.array([10., 20., 30., 40.])
    expected = 1. / x_raw

    with tfe.protocol.Pond() as prot:

      x = prot.define_constant(x_raw)
      y = prot.reciprocal(x)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        actual = sess.run(y)

      np.testing.assert_array_almost_equal(actual, expected, decimal=3)


class TestShare(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def _core_test_sharing(self, dtype):

    expected = np.array([[1, 2, 3], [4, 5, 6]])

    with tfe.protocol.Pond() as prot:

      with tfe.Session() as sess:
        shares = prot._share(dtype.tensor(expected))
        actual = sess.run(prot._reconstruct(*shares).to_native())

    np.testing.assert_array_equal(actual, expected)

  def test_int64(self):
    self._core_test_sharing(int64factory)

  def test_int100(self):
    self._core_test_sharing(int100factory)

  def test_prime(self):
    self._core_test_sharing(native_factory(tf.int32, 67))


class TestMasked(unittest.TestCase):

  def _setup(self, dtype):
    prot = tfe.protocol.Pond()
    plain_tensor = dtype.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    # pylint: disable=protected-access
    unmasked = prot._share_and_wrap(plain_tensor, False)
    # pylint: enable=protected-access
    a0 = dtype.sample_uniform(plain_tensor.shape)
    a1 = dtype.sample_uniform(plain_tensor.shape)
    a = a0 + a1
    alpha = plain_tensor - a
    x = PondMaskedTensor(self, unmasked, a, a0, a1, alpha, alpha, False)
    return prot, x

  def test_transpose_masked(self):

    with tf.Graph().as_default():

      prot, x = self._setup(int64factory)
      transpose = prot.transpose(x)

      with tfe.Session() as sess:
        actual = sess.run(transpose.reveal().to_native())
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(actual, expected)

  def test_indexer(self):

    with tf.Graph().as_default():

      prot, x = self._setup(int64factory)
      indexed = _indexer_masked(prot, x, 0)

      with tfe.Session() as sess:
        actual = sess.run(indexed.reveal().to_native())
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(actual, expected)

  def test_gather(self):

    with tf.Graph().as_default():

      prot, x = self._setup(int64factory)
      gathered = _gather_masked(prot, x, [0, 2], axis=1)

      with tfe.Session() as sess:
        actual = sess.run(gathered.reveal().to_native())
        expected = np.array([[1, 3], [4, 6]])
        np.testing.assert_array_equal(actual, expected)

  def test_negative_masked(self):

    with tf.Graph().as_default():

      prot, x = self._setup(int64factory)
      negative = _negative_masked(prot, x)

      with tfe.Session() as sess:
        actual = sess.run(negative.reveal().to_native())
        expected = np.array([[-1, -2, -3], [-4, -5, -6]])
        np.testing.assert_array_equal(actual, expected)


class TestIdentity(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def test_same_value_different_instance(self):

    expected = np.array([[1, 2, 3], [4, 5, 6]])

    with tfe.protocol.Pond() as prot:

      x = prot.define_private_variable(expected)
      y = prot.identity(x)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        actual = sess.run(y.reveal())

    assert x is not y
    np.testing.assert_array_equal(actual, expected)


class TestPondAssign(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

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

      return tfe.protocol.pond.PondPrivateTensor(prot, z_shares[0], z_shares[1],
                                                 x.is_scaled)

    a = prot.define_private_variable(tf.ones(shape=(1, 1)))
    b = prot.define_private_variable(tf.ones(shape=(1, 1)))

    op = prot.assign(a, poc(a, b))

    with tfe.Session() as sess:
      sess.run(tfe.global_variables_initializer())

      for _ in range(100):
        sess.run(op)

      result = sess.run(a.reveal())
      assert result == np.array([101.])

  def test_public_assign(self):

    with tfe.protocol.Pond() as prot:
      x_var = prot.define_public_variable(np.zeros(shape=(2, 2)))
      data = np.ones((2, 2))
      x_pl = tfe.define_public_placeholder(shape=(2, 2))
      fd = x_pl.feed(data.reshape((2, 2)))

      with tfe.Session() as sess:
        sess.run(tfe.assign(x_var, x_pl), feed_dict=fd)
        result = sess.run(x_var)
        np.testing.assert_array_equal(result, np.ones([2, 2]))


if __name__ == '__main__':
  unittest.main()
