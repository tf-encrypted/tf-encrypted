# pylint: disable=missing-docstring
import logging
from typing import List, Tuple
import os
import unittest
import pytest

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten, Conv2D, Dense
from tensorflow.keras.models import Sequential

import tf_encrypted as tfe
from tf_encrypted.convert import Converter
from tf_encrypted.convert.register import registry


_GLOBAL_FILENAME = ''
_SEED = 826485786
np.random.seed(_SEED)
tf.set_random_seed(_SEED)


class TestConvert(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

    self.previous_logging_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)

  def tearDown(self):
    global _GLOBAL_FILENAME

    tf.reset_default_graph()
    K.clear_session()

    logging.debug("Cleaning file: %s", _GLOBAL_FILENAME)
    os.remove(_GLOBAL_FILENAME)

    logging.getLogger().setLevel(self.previous_logging_level)

  @staticmethod
  def ndarray_input_fn(x):
    def input_fn():
      return tf.constant(x)
    return input_fn

  @staticmethod
  def _assert_successful_conversion(
      prot,
      graph_def,
      actual,
      *input_fns,
      decimals=3,
      **kwargs  # pylint: disable=unused-argument
  ):
    prot.clear_initializers()
    converter = Converter(
        registry(),
        config=tfe.get_config(),
        protocol=prot,
        model_provider='model-provider',
    )
    x = converter.convert(graph_def, 'input-provider', list(input_fns))

    with tfe.Session() as sess:
      sess.run(tf.global_variables_initializer())
      if not isinstance(x, (list, tuple)):
        x = [x]
        actual = [actual]
      else:
        assert isinstance(actual, (list, tuple)
                          ), "expected output to be tensor sequence"
      try:
        output = sess.run([xi.reveal().decode() for xi in x], tag='reveal')
      except AttributeError:
        # assume all xi are all public
        output = sess.run([xi for xi in x], tag='reveal')
      for o_i, a_i in zip(output, actual):
        np.testing.assert_array_almost_equal(o_i, a_i, decimal=decimals)

  @staticmethod
  def _construct_conversion_test(op_name, *test_inputs, **kwargs):
    global _GLOBAL_FILENAME
    _GLOBAL_FILENAME = '{}.pb'.format(op_name)
    exporter = globals()['export_{}'.format(op_name)]
    runner = globals()['run_{}'.format(op_name)]
    protocol = kwargs.pop('protocol')

    path = exporter(_GLOBAL_FILENAME, test_inputs[0].shape, **kwargs)
    tf.reset_default_graph()

    graph_def = read_graph(path)
    tf.reset_default_graph()

    actual = runner(*test_inputs, **kwargs)
    tf.reset_default_graph()

    prot_class = getattr(tfe.protocol, protocol)

    return graph_def, actual, prot_class

  @classmethod
  def _test_with_ndarray_input_fn(
      cls,
      op_name,
      test_input,
      protocol='Pond',
      decimals=3,
      **kwargs
  ):
    # Treat as an example of how to run tests with a particular kind of input
    graph_def, actual, prot_class = cls._construct_conversion_test(
        op_name,
        test_input,
        protocol=protocol,
        **kwargs)
    with prot_class() as prot:
      input_fn = cls.ndarray_input_fn(test_input)
      cls._assert_successful_conversion(
          prot, graph_def, actual, input_fn, decimals=decimals, **kwargs)

  def test_keras_multilayer(self):
    test_input = np.ones([1, 8, 8, 1])
    self._test_with_ndarray_input_fn(
        'keras_multilayer', test_input, protocol='SecureNN',
    )

  def test_conv2d_convert(self):
    test_input = np.ones([1, 1, 8, 8])
    self._test_with_ndarray_input_fn('conv2d', test_input, protocol='Pond')

    test_input = np.ones([1, 8, 8, 1])
    self._test_with_ndarray_input_fn(
        'conv2d', test_input, protocol='Pond', data_format='NHWC')

  def test_matmul_convert(self):
    test_input = np.ones([1, 28])
    self._test_with_ndarray_input_fn('matmul', test_input, protocol='Pond')

  def test_neg_convert(self):
    test_input = np.ones([2, 2])
    self._test_with_ndarray_input_fn('neg', test_input, protocol='Pond')

  def test_add_convert(self):
    test_input = np.ones([28, 1])
    self._test_with_ndarray_input_fn('add', test_input, protocol='Pond')

  def test_transpose_convert(self):
    test_input = np.ones([1, 2, 3, 4])
    self._test_with_ndarray_input_fn('transpose', test_input, protocol='Pond')

  def test_reshape_convert(self):
    test_input = np.ones([1, 2, 3, 4])
    self._test_with_ndarray_input_fn('reshape', test_input, protocol='Pond')

  def test_expand_dims_convert(self):
    test_input = np.ones([2, 3, 4])
    self._test_with_ndarray_input_fn(
        'expand_dims', test_input, protocol='Pond')

  def test_pad_convert(self):
    test_input = np.ones([2, 3])
    self._test_with_ndarray_input_fn('pad', test_input, protocol='Pond')

  def test_batch_to_space_nd_convert(self):
    test_input = np.ones([8, 1, 3, 1])
    self._test_with_ndarray_input_fn(
        'batch_to_space_nd', test_input, protocol='Pond')

  def test_space_to_batch_nd_convert(self):
    test_input = np.ones([2, 2, 4, 1])
    self._test_with_ndarray_input_fn(
        'space_to_batch_nd', test_input, protocol='Pond')

  def test_squeeze_convert(self):
    test_input = np.ones([1, 2, 3, 1])
    self._test_with_ndarray_input_fn('squeeze', test_input, protocol='Pond')

  def test_split_convert(self):
    test_input = np.random.random([1, 10, 10, 3])
    self._test_with_ndarray_input_fn('split', test_input, protocol='Pond')

  def test_split_v_convert(self):
    test_input = np.random.random([1, 10, 10, 3])
    self._test_with_ndarray_input_fn('split_v', test_input, protocol='Pond')

  def test_concat_convert(self):
    test_input = np.ones([1, 10, 10, 3])
    self._test_with_ndarray_input_fn('concat', test_input, protocol='Pond')

  def test_sub_convert(self):
    test_input = np.ones([28, 1])
    self._test_with_ndarray_input_fn('sub', test_input, protocol='Pond')

  def test_mul_convert(self):
    test_input = np.array([[1., 2., 3., 4.]])
    self._test_with_ndarray_input_fn('mul', test_input, protocol='Pond')

  def test_strided_slice_convert(self):
    test_input = np.ones((3, 2, 3))
    # test_input = np.array([[[1., 1., 1.], [2., 2., 2.]],
    #                        [[3., 3., 3.], [4., 4., 4.]],
    #                        [[5., 5., 5.], [6., 6., 6.]]])
    self._test_with_ndarray_input_fn(
        'strided_slice', test_input, protocol='Pond')

  def test_slice_convert(self):
    test_input = np.array([[[1., 1., 1.], [2., 2., 2.]],
                           [[3., 3., 3.], [4., 4., 4.]],
                           [[5., 5., 5.], [6., 6., 6.]]])
    self._test_with_ndarray_input_fn('slice', test_input, protocol='Pond')

  def test_batchnorm_convert(self):
    test_input = np.ones([1, 1, 28, 28])
    self._test_with_ndarray_input_fn('batchnorm', test_input, protocol='Pond')

  def test_avgpool_convert(self):
    test_input = np.ones([1, 28, 28, 1])
    self._test_with_ndarray_input_fn('avgpool', test_input, protocol='Pond')

  @pytest.mark.convert_maxpool
  def test_maxpool_convert(self):
    test_input = np.ones([1, 4, 4, 1])
    self._test_with_ndarray_input_fn(
        'maxpool', test_input, protocol='SecureNN')

  def test_stack_convert(self):
    input1 = np.array([1, 4])
    input2 = np.array([2, 5])
    input3 = np.array([3, 6])
    test_inputs = [input1, input2, input3]
    graph_def, actual, prot_class = self._construct_conversion_test(
        'stack',
        *test_inputs,
        protocol='Pond')

    with prot_class() as prot:
      input_fns = [self.ndarray_input_fn(x) for x in test_inputs]
      self._assert_successful_conversion(prot, graph_def, actual, *input_fns)

  @unittest.skipUnless(tfe.config.tensorflow_supports_int64(),
                       "Too slow on Circle CI otherwise")
  def test_argmax_convert(self):
    test_input = np.array([1., 2., 3., 4.])
    self._test_with_ndarray_input_fn(
        'argmax', test_input, protocol='SecureNN', axis=0)

  def test_required_space_to_batch_paddings_convert(self):
    test_input = np.array([4, 1, 3], dtype=np.int32)
    self._test_with_ndarray_input_fn(
        'required_space_to_batch_paddings', test_input, protocol='Pond')

  def test_flatten_convert(self):
    test_input = np.random.uniform(size=(1, 3, 3, 2)).astype(np.float32)
    self._test_with_ndarray_input_fn(
        'flatten', test_input, decimals=2, protocol='Pond')

  def test_keras_conv2d_convert(self):
    test_input = np.ones([1, 8, 8, 1])
    self._test_with_ndarray_input_fn(
        'keras_conv2d', test_input, protocol='Pond')

  def test_keras_dense_convert(self):
    test_input = np.ones([2, 10])
    self._test_with_ndarray_input_fn(
        'keras_dense',
        test_input,
        decimals=2,
        protocol='Pond',
    )

  def test_keras_batchnorm_convert(self):
    test_input = np.ones([1, 28, 28, 1])
    self._test_with_ndarray_input_fn('keras_batchnorm',
                                     test_input,
                                     protocol='Pond')


def export_argmax(filename, input_shape, axis):
  pl = tf.placeholder(tf.float32, shape=input_shape)

  output = tf.argmax(pl, axis)

  return export(output, filename)


def run_argmax(data, axis):
  inp = tf.constant(data)

  output = tf.argmax(inp, axis)

  with tf.Session() as sess:
    out = sess.run(output)

  return out


def run_stack(input1, input2, input3):
  x = tf.constant(input1)
  y = tf.constant(input2)
  z = tf.constant(input3)
  out = tf.stack([x, y, z])

  with tf.Session() as sess:
    out = sess.run(out)

  return out


def export_stack(filename: str, input_shape: Tuple[int]):
  x = tf.placeholder(tf.float32, shape=input_shape)
  y = tf.placeholder(tf.float32, shape=input_shape)
  z = tf.placeholder(tf.float32, shape=input_shape)

  out = tf.stack([x, y, z])

  return export(out, filename)


def run_avgpool(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")

  x = tf.nn.avg_pool(a, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})

  return output


def export_avgpool(filename, input_shape):
  pl = tf.placeholder(tf.float32, shape=input_shape, name="input")

  x = tf.nn.avg_pool(pl, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

  return export(x, filename)


def run_maxpool(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")

  x = tf.nn.max_pool(a, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})

  return output


def export_maxpool(filename, input_shape):
  pl = tf.placeholder(tf.float32, shape=input_shape, name="input")

  x = tf.nn.max_pool(pl, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

  return export(x, filename)


def run_batchnorm(data):
  x = tf.placeholder(tf.float32, shape=data.shape, name="input")

  dim = data.shape[3]
  mean = np.ones((1, 1, 1, dim)) * 1
  variance = np.ones((1, 1, 1, dim)) * 2
  offset = np.ones((1, 1, 1, dim)) * 3
  scale = np.ones((1, 1, 1, dim)) * 4

  y = tf.nn.batch_normalization(x, mean, variance, offset, scale, 0.00001)

  with tf.Session() as sess:
    output = sess.run(y, feed_dict={x: data})

  return output


def export_batchnorm(filename: str, input_shape: List[int]):
  pl = tf.placeholder(tf.float32, shape=input_shape, name="input")

  mean = np.ones((1, 1, 1, input_shape[3]), dtype=np.float32) * 1
  variance = np.ones((1, 1, 1, input_shape[3]), dtype=np.float32) * 2
  offset = np.ones((1, 1, 1, input_shape[3]), dtype=np.float32) * 3
  scale = np.ones((1, 1, 1, input_shape[3]), dtype=np.float32) * 4

  x = tf.nn.batch_normalization(pl, mean, variance, offset, scale, 0.00001)

  return export(x, filename)


def run_conv2d(data, data_format="NCHW"):
  feed_me = tf.placeholder(tf.float32, shape=data.shape, name="input")

  x = feed_me
  if data_format == "NCHW":
    x = tf.transpose(x, (0, 2, 3, 1))

  filtered = tf.constant(np.ones((3, 3, 1, 3)),
                         dtype=tf.float32,
                         name="weights")
  x = tf.nn.conv2d(x, filtered, (1, 1, 1, 1), "SAME", name="nn_conv2d")

  with tf.Session() as sess:
    output = sess.run(x, feed_dict={feed_me: data})

    if data_format == "NCHW":
      output = output.transpose(0, 3, 1, 2)

  return output


def export_conv2d(filename: str, input_shape: List[int], data_format="NCHW"):
  pl = tf.placeholder(tf.float32, shape=input_shape, name="input")

  filtered = tf.constant(np.ones((3, 3, 1, 3)),
                         dtype=tf.float32,
                         name="weights")
  x = tf.nn.conv2d(pl, filtered, (1, 1, 1, 1), "SAME",
                   data_format=data_format, name="nn_conv2d")

  return export(x, filename)


def run_matmul(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")
  b = tf.constant(np.ones((data.shape[1], 1)), dtype=tf.float32)

  x = tf.matmul(a, b)

  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})

  return output


def export_matmul(filename: str, input_shape: List[int]):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")
  b = tf.constant(np.ones((input_shape[1], 1)), dtype=tf.float32)

  x = tf.matmul(a, b)

  return export(x, filename)


def export_neg(filename: str, input_shape: List[int]):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")

  x = tf.negative(a)

  return export(x, filename)


def run_neg(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")

  x = tf.negative(a)

  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})

  return output


def run_add(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")
  b = tf.constant(np.ones((data.shape[1], 1)), dtype=tf.float32)

  x = tf.add(a, b)

  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})

  return output


def export_add(filename: str, input_shape: List[int]):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")
  b = tf.constant(np.ones((input_shape[0], 1)), dtype=tf.float32)

  x = tf.add(a, b)

  return export(x, filename)


def run_transpose(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")

  x = tf.transpose(a, perm=(0, 3, 1, 2))

  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})

  return output


def export_transpose(filename: str, input_shape: List[int]):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")

  x = tf.transpose(a, perm=(0, 3, 1, 2))

  return export(x, filename)


def run_reshape(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")

  last_size = 1
  for i in data.shape[1:]:
    last_size *= i

  x = tf.reshape(a, [-1, last_size])

  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})

  return output


def export_reshape(filename: str, input_shape: List[int]):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")

  last_size = 1
  for i in input_shape[1:]:
    last_size *= i

  x = tf.reshape(a, [-1, last_size])

  return export(x, filename)


def run_expand_dims(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")

  x = tf.expand_dims(a, axis=0)

  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})

  return output


def export_expand_dims(filename: str, input_shape: List[int]):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")

  x = tf.expand_dims(a, axis=0)

  return export(x, filename)


def run_pad(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")

  x = tf.pad(a, paddings=tf.constant([[2, 2], [3, 4]]), mode="CONSTANT")

  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})

  return output


def export_pad(filename: str, input_shape: List[int]):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")

  x = tf.pad(a, paddings=tf.constant([[2, 2], [3, 4]]), mode="CONSTANT")

  return export(x, filename)


def _construct_batch_to_space_nd(input_shape):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")
  block_shape = tf.constant([2, 2], dtype=tf.int32)
  crops = tf.constant([[0, 0], [2, 0]], dtype=tf.int32)
  x = tf.batch_to_space_nd(a, block_shape=block_shape, crops=crops)
  return x, a


def export_batch_to_space_nd(filename, input_shape):
  x, _ = _construct_batch_to_space_nd(input_shape)
  return export(x, filename)


def run_batch_to_space_nd(data):
  x, a = _construct_batch_to_space_nd(data.shape)
  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})
  return output


def _construct_space_to_batch_nd(input_shape):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")
  block_shape = tf.constant([2, 2], dtype=tf.int32)
  paddings = tf.constant([[0, 0], [2, 0]], dtype=tf.int32)
  x = tf.space_to_batch_nd(a, block_shape=block_shape, paddings=paddings)
  return x, a


def export_space_to_batch_nd(filename, input_shape):
  x, _ = _construct_space_to_batch_nd(input_shape)
  return export(x, filename)


def run_space_to_batch_nd(data):
  x, a = _construct_space_to_batch_nd(data.shape)
  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})
  return output


def run_squeeze(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")
  x = tf.squeeze(a, axis=[0, 3])
  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})
  return output


def export_squeeze(filename, input_shape):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")
  x = tf.squeeze(a, axis=[0, 3])
  return export(x, filename)


def run_gather(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")
  x = tf.gather(a, indices=[1, 3], axis=0)
  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})
  return output


def export_gather(filename, input_shape):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")
  x = tf.gather(a, indices=[1, 3], axis=0)
  return export(x, filename)


def run_split(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")
  x = tf.split(a, num_or_size_splits=3, axis=-1)
  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})
  return output


def export_split(filename, input_shape):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")
  x = tf.split(a, num_or_size_splits=3, axis=-1)
  return export(x, filename)


def run_split_v(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")
  x = tf.split(a, num_or_size_splits=[1, 1, 1], axis=-1)
  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})
  return output


def export_split_v(filename, input_shape):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")
  x = tf.split(a, num_or_size_splits=[1, 1, 1], axis=-1)
  return export(x, filename)


def run_concat(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")
  x = tf.concat([a, a, a], axis=-1)
  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})
  return output


def export_concat(filename, input_shape):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")
  x = tf.concat([a, a, a], axis=-1)
  return export(x, filename)


def run_sub(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")
  b = tf.constant(np.ones((data.shape[0], 1)), dtype=tf.float32)

  x = tf.subtract(a, b)

  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})

  return output


def export_sub(filename, input_shape):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")
  b = tf.constant(np.ones((input_shape[0], 1)), dtype=tf.float32)

  x = tf.subtract(a, b)

  return export(x, filename)


def run_mul(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")
  b = tf.constant(np.array([1.0, 2.0, 3.0, 4.0]).reshape(
      data.shape), dtype=tf.float32)

  x = tf.multiply(a, b)

  with tf.Session() as sess:
    output = sess.run(x, feed_dict={a: data})

  return output


def export_mul(filename, input_shape):
  a = tf.placeholder(tf.float32, shape=input_shape, name="input")
  b = tf.constant(np.array([1.0, 2.0, 3.0, 4.0]).reshape(
      input_shape), dtype=tf.float32)

  x = tf.multiply(a, b)

  return export(x, filename)


def export_strided_slice(filename, input_shape):
  t = tf.placeholder(tf.float32, shape=input_shape, name="input")
  out = tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])

  return export(out, filename)


def run_strided_slice(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")
  out = tf.strided_slice(a, [1, 0, 0], [2, 1, 3], [1, 1, 1])

  with tf.Session() as sess:
    output = sess.run(out, feed_dict={a: data})

  return output


def export_slice(filename, input_shape):
  t = tf.placeholder(tf.float32, shape=input_shape, name="input")
  out = tf.slice(t, [1, 0, 0], [2, 1, -1])

  return export(out, filename)


def run_slice(data):
  a = tf.placeholder(tf.float32, shape=data.shape, name="input")
  out = tf.slice(a, [1, 0, 0], [2, 1, -1])

  with tf.Session() as sess:
    output = sess.run(out, feed_dict={a: data})

  return output


def export_flatten(filename, input_shape):
  model = Sequential()
  model.add(Flatten(input_shape=input_shape[1:]))
  model.predict(np.random.uniform(size=input_shape))

  sess = K.get_session()
  output = model.get_layer('flatten').output

  return export(output, filename, sess=sess)


def run_flatten(data):
  model = Sequential()
  model.add(Flatten(input_shape=data.shape[1:]))
  return model.predict(data)


def keras_multilayer_builder(input_shape,
                             filters=2,
                             kernel_size=3,
                             pool_size=2,
                             units=2):
  x = tf.keras.Input(shape=input_shape[1:])
  y = tf.keras.layers.Conv2D(filters, kernel_size)(x)
  y = tf.keras.layers.ReLU()(y)
  y = tf.keras.layers.MaxPooling2D(pool_size)(y)
  y = tf.keras.layers.Flatten()(y)
  y = tf.keras.layers.Dense(units)(y)

  return tf.keras.Model(x, y)


def export_keras_multilayer(filename, input_shape):
  model, _ = _keras_model_core(keras_multilayer_builder, shape=input_shape)

  sess = K.get_session()
  output = model.output
  return export(output, filename, sess=sess)


def run_keras_multilayer(data):
  _, out = _keras_model_core(keras_multilayer_builder, data=data)
  return out


def _keras_model_core(model_builder,
                      shape=None,
                      data=None,
                      **model_builder_kwargs):
  assert shape is None or data is None
  if shape is None:
    shape = data.shape

  model = model_builder(shape, **model_builder_kwargs)

  if data is None:
    data = np.random.uniform(size=shape)
  out = model.predict(data)
  return model, out


def export_keras_conv2d(filename, input_shape):
  model, _ = _keras_conv2d_core(shape=input_shape)

  sess = K.get_session()
  output = model.get_layer('conv2d').output
  return export(output, filename, sess=sess)


def run_keras_conv2d(data):
  _, out = _keras_conv2d_core(data=data)
  return out


def _keras_conv2d_core(shape=None, data=None):
  assert shape is None or data is None
  if shape is None:
    shape = data.shape

  model = Sequential()
  c2d = Conv2D(2, (3, 3),
               data_format="channels_last",
               use_bias=False,
               input_shape=shape[1:])
  model.add(c2d)

  if data is None:
    data = np.random.uniform(size=shape)
  out = model.predict(data)
  return model, out


def export_keras_dense(filename, input_shape):
  model, _ = _keras_dense_core(shape=input_shape)

  sess = K.get_session()
  output = model.get_layer('dense').output
  return export(output, filename, sess=sess)


def run_keras_dense(data):
  _, out = _keras_dense_core(data=data)
  return out


def _keras_dense_core(shape=None, data=None):
  assert shape is None or data is None
  if shape is None:
    shape = data.shape

  model = Sequential()
  d = Dense(2,
            use_bias=True,
            input_shape=shape[1:])
  model.add(d)

  if data is None:
    data = np.random.uniform(size=shape)
  out = model.predict(data)
  return model, out

def export_keras_batchnorm(filename, input_shape):
  model, _ = _keras_batchnorm_core(shape=input_shape)

  sess = K.get_session()
  output = model.get_layer('batch_normalization_v1').output
  return export(output, filename, sess=sess)


def run_keras_batchnorm(data):
  _, out = _keras_batchnorm_core(data=data)
  return out


def _keras_batchnorm_core(shape=None, data=None):
  assert shape is None or data is None
  if shape is None:
    shape = data.shape

  model = Sequential()
  bn = tf.keras.layers.BatchNormalization(input_shape=shape[1:])
  model.add(bn)

  if data is None:
    data = np.random.uniform(size=shape)
  out = model.predict(data)
  return model, out


def run_required_space_to_batch_paddings(data):

  x = tf.placeholder(tf.int32, shape=data.shape, name="input_shape")
  y = tf.constant(np.array([2, 3, 2]), dtype=tf.int32)
  p = tf.constant(np.array([[2, 3], [4, 3], [5, 2]]), dtype=tf.int32)

  out = tf.required_space_to_batch_paddings(x, y, base_paddings=p)

  with tf.Session() as sess:
    output = sess.run(out, feed_dict={x: data})

  return output


def export_required_space_to_batch_paddings(filename, input_shape):

  x = tf.placeholder(tf.int32, shape=input_shape, name="input_shape")
  y = tf.constant(np.array([2, 3, 2]), dtype=tf.int32)
  p = tf.constant(np.array([[2, 3], [4, 3], [5, 2]]), dtype=tf.int32)

  out = tf.required_space_to_batch_paddings(x, y, base_paddings=p)

  return export(out, filename)


def export(x: tf.Tensor, filename: str, sess=None):
  should_close = False
  if sess is None:
    should_close = True
    sess = tf.Session()

  pred_node_names = ["output"]
  tf.identity(x, name=pred_node_names[0])
  graph = graph_util.convert_variables_to_constants(
      sess,
      sess.graph.as_graph_def(),
      pred_node_names
  )

  graph = graph_util.remove_training_nodes(graph)

  path = graph_io.write_graph(graph, ".", filename, as_text=False)

  if should_close:
    sess.close()

  return path


def read_graph(path: str):
  with gfile.GFile(path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  return graph_def


if __name__ == '__main__':
  unittest.main()
