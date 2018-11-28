import unittest
import os
import logging

from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.convert import Converter
from tf_encrypted.convert.register import register

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io


global_filename = ''


class TestConvert(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def tearDown(self):
        global global_filename

        logging.debug("Cleaning file: %s" % global_filename)
        os.remove(global_filename)

    @staticmethod
    def ndarray_input_fn(x):
        def input_fn():
            return tf.constant(x)
        return input_fn

    @staticmethod
    def _assert_successful_conversion(prot, graph_def, actual, *input_fns, **kwargs):
        prot.clear_initializers()

        converter = Converter(tfe.get_config(), prot, 'model-provider')
        x = converter.convert(graph_def, register(), 'input-provider', list(input_fns))

        with tfe.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    @staticmethod
    def _construct_conversion_test(op_name, protocol='Pond', *test_inputs, **kwargs):
        global global_filename
        global_filename = '{}.pb'.format(op_name)
        exporter = globals()['export_{}'.format(op_name)]
        runner = globals()['run_{}'.format(op_name)]

        path = exporter(global_filename, test_inputs[0].shape, **kwargs)
        tf.reset_default_graph()

        graph_def = read_graph(path)
        tf.reset_default_graph()

        actual = runner(*test_inputs, **kwargs)
        tf.reset_default_graph()

        prot_class = getattr(tfe.protocol, protocol)

        return graph_def, actual, prot_class

    @classmethod
    def _test_with_ndarray_input_fn(cls, op_name, test_input, protocol='Pond', **kwargs):
        # Treat this as an example of how to run tests with a particular kind of input
        graph_def, actual, prot_class = cls._construct_conversion_test(op_name,
                                                                       test_input,
                                                                       protocol,
                                                                       **kwargs)
        with prot_class() as prot:
            input_fn = cls.ndarray_input_fn(test_input)
            cls._assert_successful_conversion(prot, graph_def, actual, input_fn, **kwargs)

    def test_cnn_convert(self):
        test_input = np.ones([1, 1, 28, 28])
        self._test_with_ndarray_input_fn('cnn', 'Pond', test_input)

        test_input = np.ones([1, 28, 28, 1])
        self._test_with_ndarray_input_fn('cnn', 'Pond', test_input, data_format='NHWC')

    def test_matmul_convert(self):
        test_input = np.ones([1, 28])
        self._test_with_ndarray_input_fn('matmul', 'Pond', test_input)

    def test_add_convert(self):
        test_input = np.ones([28, 1])
        self._test_with_ndarray_input_fn('add', 'Pond', test_input)

    def test_transpose_convert(self):
        test_input = np.ones([1, 2, 3, 4])
        self._test_with_ndarray_input_fn('transpose', 'Pond', test_input)

    def test_reshape_convert(self):
        test_input = np.ones([1, 2, 3, 4])
        self._test_with_ndarray_input_fn('reshape', 'Pond', test_input)

    def test_expand_dims_convert(self):
        test_input = np.ones([2, 3, 4])
        self._test_with_ndarray_input_fn('expand_dims', 'Pond', test_input)

    def test_pad_convert(self):
        test_input = np.ones([2, 3])
        self._test_with_ndarray_input_fn('pad', 'Pond', test_input)

    def test_batch_to_space_nd_convert(self):
        test_input = np.ones([8, 1, 3, 1])
        self._test_with_ndarray_input_fn('batch_to_space_nd', 'Pond', test_input)

    def test_space_to_batch_nd_convert(self):
        test_input = np.ones([2, 2, 4, 1])
        self._test_with_ndarray_input_fn('space_to_batch_nd', 'Pond', test_input)

    def test_batch_to_space_nd_convert(self):
        global global_filename
        global_filename = "batch_to_space_nd.pb"
        input_shape = [8, 1, 3, 1]

        path = export_batch_to_space_nd(global_filename, input_shape)
        tf.reset_default_graph()
        graph_def = read_graph(path)
        tf.reset_default_graph()
        actual = run_batch_to_space_nd(input_shape)
        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))
            converter = Converter(tfe.get_config(), prot, 'model-provider')
            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_space_to_batch_nd_convert(self):
        global global_filename
        global_filename = "space_to_batch_nd.pb"
        input_shape = [2, 2, 4, 1]
        path = export_space_to_batch_nd(global_filename, input_shape)
        tf.reset_default_graph()

        graph_def = read_graph(path)
        tf.reset_default_graph()

        actual = run_space_to_batch_nd(input_shape)
        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))
            converter = Converter(tfe.get_config(), prot, 'model-provider')
            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_squeeze_convert(self):
        test_input = np.ones([1, 2, 3, 1])
        self._test_with_ndarray_input_fn('squeeze', 'Pond', test_input)

    def test_sub_convert(self):
        test_input = no.ones([28, 1])
        self._test_with_ndarray_input_fn('sub', 'Pond', test_input)

    def test_mul_convert(self):
        test_input = np.array([[1., 2., 3., 4.]])
        self._test_with_ndarray_input_fn('mul', 'Pond', test_input)

    def test_strided_slice_convert(self):
        test_input = np.array([[[1, 1, 1], [2, 2, 2]],
                               [[3, 3, 3], [4, 4, 4]],
                               [[5, 5, 5], [6, 6, 6]]])
        self._test_with_ndarray_input_fn('strided_slice', 'Pond', test_input)

    def test_batchnorm_convert(self):
        test_input = np.ones([1, 1, 28, 28])
        self._test_with_ndarray_input_fn('batchnorm', 'Pond', test_input)

    def test_avgpool_convert(self):
        test_input = np.ones([1, 28, 28, 1])
        self._test_with_ndarray_input_fn('avgpooling', 'Pond', test_input)

    def test_maxpool_convert(self):
        test_input = np.ones([1, 28, 28, 1])
        self._test_with_ndarray_input_fn('maxpool', 'SecureNN', test_input)

    def test_stack_convert(self):
        input1 = np.array([1, 4])
        input2 = np.array([2, 5])
        input3 = np.array([3, 6])
        graph_def, actual, prot_class = cls._construct_conversion_test('stack',
                                                                       'Pond',
                                                                       input1,
                                                                       input2,
                                                                       input3,
                                                                       **kwargs)

        with prot_class() as prot:
            input_fns = [self.ndarray_input_fn(x) for x in test_inputs]
            self._assert_successful_conversion(prot, graph_def, actual, *input_fns, **kwargs)


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


def run_avgpool(input):
    input = tf.placeholder(tf.float32, shape=input.shape, name="input")

    x = tf.nn.avg_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={input: input})

    return output


def export_avgpool(filename, input_shape):
    input = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.nn.avg_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    return export(x, filename)


def run_maxpool(input):
    input = tf.placeholder(tf.float32, shape=input.shape, name="input")

    x = tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={input: input})

    return output


def export_maxpool(filename, input_shape):
    input = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    return export(x, filename)


def run_batchnorm(input):
    x = tf.placeholder(tf.float32, shape=input.shape, name="input")

    dim = input.shape[3]
    mean = np.ones((1, 1, 1, dim)) * 1
    variance = np.ones((1, 1, 1, dim)) * 2
    offset = np.ones((1, 1, 1, dim)) * 3
    scale = np.ones((1, 1, 1, dim)) * 4

    y = tf.nn.batch_normalization(x, mean, variance, offset, scale, 0.00001)

    with tf.Session() as sess:
        output = sess.run(y, feed_dict={x: input})

    return output


def export_batchnorm(filename: str, input_shape: List[int]):
    input = tf.placeholder(tf.float32, shape=input_shape, name="input")

    mean = np.ones((1, 1, 1, input_shape[3]), dtype=np.float32) * 1
    variance = np.ones((1, 1, 1, input_shape[3]), dtype=np.float32) * 2
    offset = np.ones((1, 1, 1, input_shape[3]), dtype=np.float32) * 3
    scale = np.ones((1, 1, 1, input_shape[3]), dtype=np.float32) * 4

    x = tf.nn.batch_normalization(input, mean, variance, offset, scale, 0.00001)

    return export(x, filename)


def run_cnn(input, data_format="NCHW"):
    feed_me = tf.placeholder(tf.float32, shape=input.shape, name="input")

    x = ph
    if data_format == "NCHW":
        x = tf.transpose(ph, (0, 2, 3, 1))

    filter = tf.constant(np.ones((5, 5, 1, 16)), dtype=tf.float32, name="weights")
    x = tf.nn.conv2d(x, filter, (1, 1, 1, 1), "SAME", name="conv2d")

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={feed_me: input})

        if data_format == "NCHW":
            output = output.transpose(0, 3, 1, 2)

    return output


def export_cnn(filename: str, input_shape: List[int], data_format="NCHW"):
    input = tf.placeholder(tf.float32, shape=input_shape, name="input")

    filter = tf.constant(np.ones((5, 5, 1, 16)), dtype=tf.float32, name="weights")
    x = tf.nn.conv2d(input, filter, (1, 1, 1, 1), "SAME", data_format=data_format, name="conv2d")

    return export(x, filename)


def run_matmul(input):
    a = tf.placeholder(tf.float32, shape=input.shape, name="input")
    b = tf.constant(np.ones((input.shape[1], 1)), dtype=tf.float32)

    x = tf.matmul(a, b)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: input})

    return output


def export_matmul(filename: str, input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    b = tf.constant(np.ones((input_shape[1], 1)), dtype=tf.float32)

    x = tf.matmul(a, b)

    return export(x, filename)


def run_add(input):
    a = tf.placeholder(tf.float32, shape=input.shape, name="input")
    b = tf.constant(np.ones((input.shape[1], 1)), dtype=tf.float32)

    x = tf.add(a, b)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: input})

    return output


def export_add(filename: str, input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    b = tf.constant(np.ones((input_shape[0], 1)), dtype=tf.float32)

    x = tf.add(a, b)

    return export(x, filename)


def run_transpose(input):
    a = tf.placeholder(tf.float32, shape=input.shape, name="input")

    x = tf.transpose(a, perm=(0, 3, 1, 2))

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: input})

    return output


def export_transpose(filename: str, input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.transpose(a, perm=(0, 3, 1, 2))

    return export(x, filename)


def run_reshape(input):
    a = tf.placeholder(tf.float32, shape=input.shape, name="input")

    last_size = 1
    for i in input.shape[1:]:
        last_size *= i

    x = tf.reshape(a, [-1, last_size])

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: input})

    return output


def export_reshape(filename: str, input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")

    last_size = 1
    for i in input_shape[1:]:
        last_size *= i

    x = tf.reshape(a, [-1, last_size])

    return export(x, filename)


def run_expand_dims(input):
    a = tf.placeholder(tf.float32, shape=input.shape, name="input")

    x = tf.expand_dims(a, axis=0)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: input})

    return output


def export_expand_dims(filename: str, input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.expand_dims(a, axis=0)

    return export(x, filename)


def run_pad(input):
    a = tf.placeholder(tf.float32, shape=input.shape, name="input")

    x = tf.pad(a, paddings=tf.constant([[2, 2], [3, 4]]), mode="CONSTANT")

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: input})

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
    return x


def export_batch_to_space_nd(filename, input_shape):
    x = _core_batch_to_space_nd_construct(input_shape)
    return export(x, filename)


def run_batch_to_space_nd(input):
    x = _core_batch_to_space_nd_construct(input.shape)
    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: input})
    return output


def _construct_space_to_batch_nd(input_shape):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    block_shape = tf.constant([2, 2], dtype=tf.int32)
    crops = tf.constant([[0, 0], [2, 0]], dtype=tf.int32)
    x = tf.space_to_batch_nd(a, block_shape=block_shape, crops=crops)
    return x


def export_space_to_batch_nd(filename, input_shape):
    x = _core_space_to_batch_nd_construct(input_shape)
    return export(x, filename)


def run_space_to_batch_nd(input):
    x = _core_space_to_batch_nd_construct(input.shape)
    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: input})
    return output


def run_squeeze(input):
    a = tf.placeholder(tf.float32, shape=input.shape, name="input")
    x = tf.squeeze(a, axis=[0, 3])
    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: input})
    return output


def export_squeeze(filename, input_shape):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    x = tf.squeeze(a, axis=[0, 3])
    return export(x, filename)


def run_sub(input):
    a = tf.placeholder(tf.float32, shape=input.shape, name="input")
    b = tf.constant(np.ones((input.shape[0], 1)), dtype=tf.float32)

    x = tf.subtract(a, b)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: input})

    return output


def export_sub(filename, input_shape):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    b = tf.constant(np.ones((input_shape[0], 1)), dtype=tf.float32)

    x = tf.subtract(a, b)

    return export(x, filename)


def run_mul(input):
    a = tf.placeholder(tf.float32, shape=input.shape, name="input")
    b = tf.constant(np.array([1.0, 2.0, 3.0, 4.0]).reshape(input.shape), dtype=tf.float32)

    x = tf.multiply(a, b)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: input})

    return output


def export_mul(filename, input_shape):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    b = tf.constant(np.array([1.0, 2.0, 3.0, 4.0]).reshape(input_shape), dtype=tf.float32)

    x = tf.multiply(a, b)

    return export(x, filename)


def export_strided_slice(filename: str, input_shape: List[int] = [3, 2, 3]):
    t = tf.placeholder(tf.float32, shape=input_shape, name="input")
    out = tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])

    return export(out, filename)


def run_strided_slice(input):
    a = tf.placeholder(tf.float32, shape=input.shape, name="input")
    out = tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])

    with tf.Session() as sess:
        output = sess.run(out).feed_dict({a: input})

    return output


def export(x: tf.Tensor, filename: str):
    with tf.Session() as sess:
        pred_node_names = ["output"]
        tf.identity(x, name=pred_node_names[0])

        graph = graph_util.convert_variables_to_constants(sess,
                                                          sess.graph.as_graph_def(),
                                                          pred_node_names)

        graph = graph_util.remove_training_nodes(graph)

        path = graph_io.write_graph(graph, ".", filename, as_text=False)

    return path


def read_graph(path: str):
    with gfile.FastGFile(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


if __name__ == '__main__':
    unittest.main()
