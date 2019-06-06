# pylint: disable=missing-docstring
import unittest
import pytest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe


class TestBatchToSpaceND(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()
    tf.set_random_seed(4224)

  def test_4d_no_crops(self):
    backing = [[[[1], [3]], [[9], [11]]],
               [[[2], [4]], [[10], [12]]],
               [[[5], [7]], [[13], [15]]],
               [[[6], [8]], [[14], [16]]]]
    t = tf.constant(backing)
    block_shape = [2, 2]
    crops = [[0, 0], [0, 0]]
    self._generic_private_test(t, block_shape, crops)

  def test_4d_single_crop(self):
    backing = [[[[0], [1], [3]]], [[[0], [9], [11]]],
               [[[0], [2], [4]]], [[[0], [10], [12]]],
               [[[0], [5], [7]]], [[[0], [13], [15]]],
               [[[0], [6], [8]]], [[[0], [14], [16]]]]
    t = tf.constant(backing)
    block_shape = [2, 2]
    crops = [[0, 0], [2, 0]]
    self._generic_private_test(t, block_shape, crops)

  def test_3d_no_crops(self):
    t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
    block_shape = [4]
    crops = [[0, 0]]
    self._generic_private_test(t, block_shape, crops)

  def test_3d_mirror_crops(self):
    t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
    block_shape = [4]
    crops = [[2, 2]]
    self._generic_private_test(t, block_shape, crops)

  def test_3d_uneven_crops(self):
    t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
    block_shape = [4]
    crops = [[2, 0]]
    self._generic_private_test(t, block_shape, crops)

  def test_3d_block_shape(self):
    t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
    block_shape = [8]
    crops = [[0, 0]]
    self._generic_private_test(t, block_shape, crops)

  def test_public(self):
    t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
    block_shape = [4]
    crops = [[2, 2]]
    self._generic_public_test(t, block_shape, crops)

  def test_masked(self):
    t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
    block_shape = [4]
    crops = [[2, 2]]
    self._generic_masked_test(t, block_shape, crops)

  @staticmethod
  def _generic_public_test(t, block_shape, crops):
    with tf.Session() as sess:
      out = tf.batch_to_space_nd(t, block_shape=block_shape, crops=crops)
      actual = sess.run(out)

    with tfe.protocol.Pond() as prot:
      b = prot.define_public_variable(t)
      out = prot.batch_to_space_nd(b, block_shape=block_shape, crops=crops)
      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final = sess.run(out)

    np.testing.assert_array_almost_equal(final, actual, decimal=3)

  @staticmethod
  def _generic_private_test(t, block_shape, crops):
    with tf.Session() as sess:
      out = tf.batch_to_space_nd(t, block_shape=block_shape, crops=crops)
      actual = sess.run(out)

    with tfe.protocol.Pond() as prot:
      b = prot.define_private_variable(t)
      out = prot.batch_to_space_nd(b, block_shape=block_shape, crops=crops)
      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final = sess.run(out.reveal())

    np.testing.assert_array_almost_equal(final, actual, decimal=3)

  @staticmethod
  def _generic_masked_test(t, block_shape, crops):
    with tf.Session() as sess:
      out = tf.batch_to_space_nd(t, block_shape=block_shape, crops=crops)
      actual = sess.run(out)

    with tfe.protocol.Pond() as prot:
      b = prot.mask(prot.define_private_variable(t))
      out = prot.batch_to_space_nd(b, block_shape=block_shape, crops=crops)
      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final = sess.run(out.reveal())

    np.testing.assert_array_almost_equal(final, actual, decimal=3)


class TestSpaceToBatchND(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()
    tf.set_random_seed(4224)

  def test_4d_no_crops(self):
    backing = [[[[1], [3]], [[9], [11]]],
               [[[2], [4]], [[10], [12]]],
               [[[5], [7]], [[13], [15]]],
               [[[6], [8]], [[14], [16]]]]
    t = tf.constant(backing)
    block_shape = [2, 2]
    paddings = [[0, 0], [0, 0]]
    self._generic_private_test(t, block_shape, paddings)

  def test_4d_single_crop(self):
    backing = [[[[1], [2], [3], [4]],
                [[5], [6], [7], [8]]],
               [[[9], [10], [11], [12]],
                [[13], [14], [15], [16]]]]
    t = tf.constant(backing)
    block_shape = [2, 2]
    paddings = [[0, 0], [2, 0]]
    self._generic_private_test(t, block_shape, paddings)

  def test_3d_no_crops(self):
    t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
    block_shape = [4]
    paddings = [[0, 0]]
    self._generic_private_test(t, block_shape, paddings)

  def test_3d_mirror_crops(self):
    t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
    block_shape = [4]
    paddings = [[2, 2]]
    self._generic_private_test(t, block_shape, paddings)

  def test_3d_uneven_crops(self):
    t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
    block_shape = [2]
    paddings = [[2, 0]]
    self._generic_private_test(t, block_shape, paddings)

  def test_3d_block_shape(self):
    t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
    block_shape = [5]
    paddings = [[0, 0]]
    self._generic_private_test(t, block_shape, paddings)

  def test_public(self):
    t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
    block_shape = [4]
    paddings = [[2, 2]]
    self._generic_public_test(t, block_shape, paddings)

  def test_masked(self):
    t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
    block_shape = [4]
    paddings = [[2, 2]]
    self._generic_masked_test(t, block_shape, paddings)

  @staticmethod
  def _generic_public_test(t, block_shape, paddings):
    with tf.Session() as sess:
      out = tf.space_to_batch_nd(t, block_shape=block_shape, paddings=paddings)
      actual = sess.run(out)

    with tfe.protocol.Pond() as prot:
      b = prot.define_public_variable(t)
      out = prot.space_to_batch_nd(
          b, block_shape=block_shape, paddings=paddings)
      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final = sess.run(out)

    np.testing.assert_array_almost_equal(final, actual, decimal=3)

  @staticmethod
  def _generic_private_test(t, block_shape, paddings):
    with tf.Session() as sess:
      out = tf.space_to_batch_nd(t, block_shape=block_shape, paddings=paddings)
      actual = sess.run(out)

    with tfe.protocol.Pond() as prot:
      b = prot.define_private_variable(t)
      out = prot.space_to_batch_nd(
          b, block_shape=block_shape, paddings=paddings)
      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final = sess.run(out.reveal())

    np.testing.assert_array_almost_equal(final, actual, decimal=3)

  @staticmethod
  def _generic_masked_test(t, block_shape, paddings):
    with tf.Session() as sess:
      out = tf.space_to_batch_nd(t, block_shape=block_shape, paddings=paddings)
      actual = sess.run(out)

    with tfe.protocol.Pond() as prot:
      b = prot.mask(prot.define_private_variable(t))
      out = prot.space_to_batch_nd(
          b, block_shape=block_shape, paddings=paddings)
      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final = sess.run(out.reveal())

    np.testing.assert_array_almost_equal(final, actual, decimal=3)



class Testconcat(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_concat(self):

    with tf.Session() as sess:
      t1 = [[1, 2, 3], [4, 5, 6]]
      t2 = [[7, 8, 9], [10, 11, 12]]
      out = tf.concat([t1, t2], 0)
      actual = sess.run(out)

    tf.reset_default_graph()

    with tfe.protocol.Pond() as prot:
      x = prot.define_private_variable(np.array(t1))
      y = prot.define_private_variable(np.array(t2))

      out = prot.concat([x, y], 0)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final = sess.run(out.reveal())

    np.testing.assert_array_equal(final, actual)

  def test_masked_concat(self):

    with tf.Session() as sess:
      t1 = [[1, 2, 3], [4, 5, 6]]
      t2 = [[7, 8, 9], [10, 11, 12]]
      out = tf.concat([t1, t2], 0)
      actual = sess.run(out)

    tf.reset_default_graph()

    with tfe.protocol.Pond() as prot:
      x = prot.mask(prot.define_private_variable(np.array(t1)))
      y = prot.mask(prot.define_private_variable(np.array(t2)))

      out = prot.concat([x, y], 0)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final = sess.run(out.unmasked.reveal())

    np.testing.assert_array_equal(final, actual)



class TestConv2D(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_forward(self) -> None:
    # input
    batch_size, channels_in, channels_out = 32, 3, 64
    img_height, img_width = 28, 28
    input_shape = (batch_size, channels_in, img_height, img_width)
    input_conv = np.random.normal(size=input_shape).astype(np.float32)

    # filters
    h_filter, w_filter, strides = 2, 2, 2
    filter_shape = (h_filter, w_filter, channels_in, channels_out)
    filter_values = np.random.normal(size=filter_shape)

    # convolution pond
    with tfe.protocol.Pond() as prot:

      conv_input = prot.define_private_variable(input_conv)
      conv_layer = tfe.layers.Conv2D(input_shape, filter_shape, strides=2)
      conv_layer.initialize(initial_weights=filter_values)
      conv_out_pond = conv_layer.forward(conv_input)

      with tfe.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # outputs
        out_pond = sess.run(conv_out_pond.reveal())

    # reset graph
    tf.reset_default_graph()

    # convolution tensorflow
    with tf.Session() as sess:
      # conv input
      x = tf.Variable(input_conv, dtype=tf.float32)
      x_nhwc = tf.transpose(x, (0, 2, 3, 1))

      # convolution Tensorflow
      filters_tf = tf.Variable(filter_values, dtype=tf.float32)

      conv_out_tf = tf.nn.conv2d(x_nhwc, filters_tf,
                                 strides=[1, strides, strides, 1],
                                 padding="SAME")

      sess.run(tf.global_variables_initializer())
      out_tensorflow = sess.run(conv_out_tf).transpose(0, 3, 1, 2)

    np.testing.assert_allclose(out_pond, out_tensorflow, atol=0.01)

  def test_forward_bias(self) -> None:
    # input
    batch_size, channels_in, channels_out = 32, 3, 64
    img_height, img_width = 28, 28
    input_shape = (batch_size, channels_in, img_height, img_width)
    input_conv = np.random.normal(size=input_shape).astype(np.float32)

    # filters
    h_filter, w_filter, strides = 2, 2, 2
    filter_shape = (h_filter, w_filter, channels_in, channels_out)
    filter_values = np.random.normal(size=filter_shape)

    # convolution pond
    with tfe.protocol.Pond() as prot:

      conv_input = prot.define_private_variable(input_conv)
      conv_layer = tfe.layers.Conv2D(input_shape, filter_shape, strides=2)

      output_shape = conv_layer.get_output_shape()

      bias = np.random.uniform(size=output_shape[1:])

      conv_layer.initialize(initial_weights=filter_values, initial_bias=bias)
      conv_out_pond = conv_layer.forward(conv_input)

      with tfe.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # outputs
        out_pond = sess.run(conv_out_pond.reveal())

    # reset graph
    tf.reset_default_graph()

    # convolution tensorflow
    with tf.Session() as sess:
      # conv input
      x = tf.Variable(input_conv, dtype=tf.float32)
      x_nhwc = tf.transpose(x, (0, 2, 3, 1))

      # convolution Tensorflow
      filters_tf = tf.Variable(filter_values, dtype=tf.float32)

      conv_out_tf = tf.nn.conv2d(x_nhwc, filters_tf,
                                 strides=[1, strides, strides, 1],
                                 padding="SAME")

      sess.run(tf.global_variables_initializer())
      out_tensorflow = sess.run(conv_out_tf).transpose(0, 3, 1, 2)
      out_tensorflow += bias

    np.testing.assert_allclose(out_pond, out_tensorflow, atol=0.01)

  def test_backward(self):
    pass



class TestMatMul(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_matmul(self) -> None:

    with tfe.protocol.Pond() as prot:

      input_shape = [4, 5]
      x_in = np.random.normal(size=input_shape)

      filter_shape = [5, 4]
      filter_values = np.random.normal(size=filter_shape)

      input_input = prot.define_private_variable(x_in)
      filter_filter = prot.define_private_variable(filter_values)

      out = prot.matmul(input_input, filter_filter)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())

        out_pond = sess.run(out.reveal())

    # reset graph
    tf.reset_default_graph()

    with tf.Session() as sess:
      x = tf.Variable(x_in, dtype=tf.float32)
      filters_tf = tf.Variable(filter_values, dtype=tf.float32)

      out = tf.matmul(x, filters_tf)

      sess.run(tf.global_variables_initializer())
      out_tensorflow = sess.run(out)

    np.testing.assert_array_almost_equal(out_pond, out_tensorflow, decimal=2)

  def test_big_middle_matmul(self) -> None:
    with tfe.protocol.Pond() as prot:

      input_shape = [64, 4500]
      x_in = np.random.normal(size=input_shape)

      filter_shape = [4500, 64]
      filter_values = np.random.normal(size=filter_shape)

      input_input = prot.define_private_variable(x_in)
      filter_filter = prot.define_private_variable(filter_values)

      out = prot.matmul(input_input, filter_filter)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())

        out_pond = sess.run(out.reveal())

    # reset graph
    tf.reset_default_graph()

    with tf.Session() as sess:
      x = tf.Variable(x_in, dtype=tf.float32)
      filters_tf = tf.Variable(filter_values, dtype=tf.float32)

      out = tf.matmul(x, filters_tf)

      sess.run(tf.global_variables_initializer())
      out_tensorflow = sess.run(out)

    np.testing.assert_allclose(out_pond, out_tensorflow, atol=.1)


class TestNegative(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_negative(self):
    input_shape = [2, 2]
    input_neg = np.ones(input_shape)

    # reshape pond
    with tfe.protocol.Pond() as prot:

      neg_input = prot.define_private_variable(input_neg)

      neg_out_pond = prot.negative(neg_input)

      with tfe.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # outputs
        out_pond = sess.run(neg_out_pond.reveal())

      # reset graph
      tf.reset_default_graph()

      with tf.Session() as sess:
        x = tf.Variable(input_neg, dtype=tf.float32)

        neg_out_tf = tf.negative(x)

        sess.run(tf.global_variables_initializer())

        out_tensorflow = sess.run(neg_out_tf)

    assert np.isclose(out_pond, out_tensorflow, atol=0.6).all()


class TestPad(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_pad(self):

    with tfe.protocol.Pond() as prot:

      tf.reset_default_graph()

      x_in = np.array([[1, 2, 3], [4, 5, 6]])
      input_input = prot.define_private_variable(x_in)

      paddings = [[2, 2], [3, 4]]

      out = prot.pad(input_input, paddings)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out_tfe = sess.run(out.reveal())

      tf.reset_default_graph()

      # TODO this is a bit weird
      out_tensorflow = tfe.convert.convert_test.run_pad(x_in)

      np.testing.assert_allclose(out_tfe, out_tensorflow, atol=.01)



@pytest.mark.slow
class TestReduceMax(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def tearDown(self):
    tf.reset_default_graph()

  def test_reduce_max_1d(self):

    t = np.array([1, 2, 3, 4]).astype(float)

    with tf.Session() as sess:
      out_tf = tf.reduce_max(t)
      expected = sess.run(out_tf)

    with tfe.protocol.SecureNN() as prot:
      b = prot.define_private_variable(tf.constant(t))
      out_tfe = prot.reduce_max(b)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(2):
          actual = sess.run(out_tfe.reveal(), tag='test_1d')

    np.testing.assert_array_equal(actual, expected)

  def test_reduce_max_2d_axis0(self):

    t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 4).astype(float)

    with tf.Session() as sess:
      out_tf = tf.reduce_max(t, axis=0)
      expected = sess.run(out_tf)

    with tfe.protocol.SecureNN() as prot:
      b = prot.define_private_variable(tf.constant(t))
      out_tfe = prot.reduce_max(b, axis=0)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(2):
          actual = sess.run(out_tfe.reveal(), tag='test_2d_axis0')

    np.testing.assert_array_equal(actual, expected)

  def test_reduce_max_2d_axis1(self):

    t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 4).astype(float)

    with tf.Session() as sess:
      out_tf = tf.reduce_max(t, axis=1)
      expected = sess.run(out_tf)

    with tfe.protocol.SecureNN() as prot:
      b = prot.define_private_variable(tf.constant(t))
      out_tfe = prot.reduce_max(b, axis=1)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(2):
          actual = sess.run(out_tfe.reveal(), tag='test_2d_axis1')

    np.testing.assert_array_equal(actual, expected)

  def test_reduce_max_3d_axis0(self):

    t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 2, 2)

    with tf.Session() as sess:
      out = tf.reduce_max(t, axis=0)
      expected = sess.run(out)

    with tfe.protocol.SecureNN() as prot:
      b = prot.define_private_variable(tf.constant(t))
      out_tfe = prot.reduce_max(b, axis=0)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(2):
          actual = sess.run(out_tfe.reveal(), tag='test_3d_axis0')

    np.testing.assert_array_equal(actual, expected)



class TestReduceSum(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_reduce_sum_1d(self):

    t = [1, 2]
    with tf.Session() as sess:
      out = tf.reduce_sum(t)
      actual = sess.run(out)

    with tfe.protocol.Pond() as prot:
      b = prot.define_private_variable(tf.constant(t))
      out = prot.reduce_sum(b)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final = sess.run(out.reveal())

    np.testing.assert_array_equal(final, actual)

  def test_reduce_sum_2d(self):

    t = [[1, 2], [1, 3]]
    with tf.Session() as sess:
      out = tf.reduce_sum(t, axis=1)
      actual = sess.run(out)

    with tfe.protocol.Pond() as prot:
      b = prot.define_private_variable(tf.constant(t))
      out = prot.reduce_sum(b, axis=1)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final = sess.run(out.reveal())

    np.testing.assert_array_equal(final, actual)

  def test_reduce_sum_huge_vector(self):

    t = [1] * 2**13
    with tf.Session() as sess:
      out = tf.reduce_sum(t)
      actual = sess.run(out)

    with tfe.protocol.Pond() as prot:
      b = prot.define_private_variable(tf.constant(t))
      out = prot.reduce_sum(b)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final = sess.run(out.reveal())

    np.testing.assert_array_equal(final, actual)


class TestStack(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_stack(self):

    with tf.Session() as sess:
      x = tf.constant([1, 4])
      y = tf.constant([2, 5])
      z = tf.constant([3, 6])
      out = tf.stack([x, y, z])

      actual = sess.run(out)

    tf.reset_default_graph()

    with tfe.protocol.Pond() as prot:
      x = prot.define_private_variable(np.array([1, 4]))
      y = prot.define_private_variable(np.array([2, 5]))
      z = prot.define_private_variable(np.array([3, 6]))

      out = prot.stack((x, y, z))

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final = sess.run(out.reveal())

    np.testing.assert_array_equal(final, actual)


class TestStridedSlice(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_strided_slice(self):

    with tf.Session() as sess:
      t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                       [[3, 3, 3], [4, 4, 4]],
                       [[5, 5, 5], [6, 6, 6]]])
      out = tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])

      actual = sess.run(out)

    tf.reset_default_graph()

    with tfe.protocol.Pond() as prot:
      x = np.array([[[1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4]],
                    [[5, 5, 5], [6, 6, 6]]])

      out = prot.define_private_variable(x)

      out = prot.strided_slice(out, [1, 0, 0], [2, 1, 3], [1, 1, 1])

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        final = sess.run(out.reveal())

    np.testing.assert_array_equal(final, actual)


if __name__ == '__main__':
  unittest.main()
