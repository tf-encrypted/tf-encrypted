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

    def test_cnn_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "cnn.pb"

        input_shape = [1, 1, 28, 28]

        path = export_cnn(global_filename, input_shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_cnn(input_shape)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_cnn_NHWC_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "cnn_nhwc.pb"

        input_shape = [1, 28, 28, 1]

        path = export_cnn(global_filename, input_shape, data_format="NHWC")

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_cnn(input_shape, data_format="NHWC")

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_matmul_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "matmul.pb"

        input_shape = [1, 28]

        path = export_matmul(global_filename, input_shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_matmul(input_shape)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_add_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "add.pb"

        input_shape = [28, 1]

        path = export_add(global_filename, input_shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_add(input_shape)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_transpose_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "transpose.pb"

        input_shape = [1, 2, 3, 4]

        path = export_transpose(global_filename, input_shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_transpose(input_shape)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_reshape_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "reshape.pb"

        input_shape = [1, 2, 3, 4]

        path = export_reshape(global_filename, input_shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_reshape(input_shape)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

            np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_expand_dims_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "expand_dims.pb"

        input_shape = [2, 3, 4]

        path = export_expand_dims(global_filename, input_shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_expand_dims(input_shape)
        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_squeeze_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "squeeze.pb"

        input_shape = [1, 2, 3, 1]

        path = export_squeeze(global_filename, input_shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_squeeze(input_shape)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_sub_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "sub.pb"

        input_shape = [28, 1]

        path = export_sub(global_filename, input_shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_sub(input_shape)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_mul_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "mul.pb"

        input_shape = [4, 1]

        path = export_mul(global_filename, input_shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_mul(input_shape)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.array([1.0, 2.0, 3.0, 4.0]).reshape(input_shape))

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_strided_slice_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "strided_slice.pb"

        path = export_strided_slice(global_filename)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        input = [[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]]

        actual = run_strided_slice(input)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant([[[1, 1, 1], [2, 2, 2]],
                                    [[3, 3, 3], [4, 4, 4]],
                                    [[5, 5, 5], [6, 6, 6]]])

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_batchnorm_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "batchnrom.pb"

        input_shape = [1, 1, 28, 28]

        path = export_batchnorm(global_filename, input_shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_batchnorm(input_shape)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_avgpooling_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "avgpool.pb"

        input_shape = [1, 28, 28, 1]

        path = export_avgpool(global_filename, input_shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_avgpool(input_shape)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_maxpooling_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "maxpool.pb"

        input_shape = [1, 28, 28, 1]

        path = export_maxpool(global_filename, input_shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_maxpool(input_shape)

        tf.reset_default_graph()

        with tfe.protocol.SecureNN() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.ones(input_shape))

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)

    def test_stack_convert(self):
        tf.reset_default_graph()

        global global_filename
        global_filename = "stack.pb"

        input1 = np.array([1, 4])
        input2 = np.array([2, 5])
        input3 = np.array([3, 6])

        path = export_stack(global_filename, input1.shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        actual = run_stack(input1, input2, input3)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input1() -> tf.Tensor:
                return tf.constant(input1)

            def provide_input2() -> tf.Tensor:
                return tf.constant(input2)

            def provide_input3() -> tf.Tensor:
                return tf.constant(input3)

            inputs = [provide_input1, provide_input2, provide_input3]

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', inputs)

            with tfe.Session() as sess:
                sess.run(prot.initializer, tag='init')

                output = sess.run(x.reveal(), tag='reveal')

        np.testing.assert_array_almost_equal(output, actual, decimal=3)


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


def run_avgpool(input_shape: List[int]):
    input = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.nn.avg_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={input: np.ones(input_shape)})

    return output


def export_avgpool(filename: str, input_shape: List[int]):
    input = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.nn.avg_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    return export(x, filename)


def run_maxpool(input_shape: List[int]):
    input = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={input: np.ones(input_shape)})

    return output


def export_maxpool(filename: str, input_shape: List[int]):
    input = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    return export(x, filename)


def run_batchnorm(input_shape: List[int]):
    input = tf.placeholder(tf.float32, shape=input_shape, name="input")

    mean = np.ones((1, 1, 1, input_shape[3])) * 1
    variance = np.ones((1, 1, 1, input_shape[3])) * 2
    offset = np.ones((1, 1, 1, input_shape[3])) * 3
    scale = np.ones((1, 1, 1, input_shape[3])) * 4

    x = tf.nn.batch_normalization(input, mean, variance, offset, scale, 0.00001)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={input: np.ones(input_shape)})

    return output


def export_batchnorm(filename: str, input_shape: List[int]):
    input = tf.placeholder(tf.float32, shape=input_shape, name="input")

    mean = np.ones((1, 1, 1, input_shape[3]), dtype=np.float32) * 1
    variance = np.ones((1, 1, 1, input_shape[3]), dtype=np.float32) * 2
    offset = np.ones((1, 1, 1, input_shape[3]), dtype=np.float32) * 3
    scale = np.ones((1, 1, 1, input_shape[3]), dtype=np.float32) * 4

    x = tf.nn.batch_normalization(input, mean, variance, offset, scale, 0.00001)

    return export(x, filename)


def run_cnn(input_shape: List[int], data_format="NCHW"):
    input = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = input
    if data_format == "NCHW":
        x = tf.transpose(input, (0, 2, 3, 1))

    filter = tf.constant(np.ones((5, 5, 1, 16)), dtype=tf.float32, name="weights")
    x = tf.nn.conv2d(x, filter, (1, 1, 1, 1), "SAME", name="conv2d")

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={input: np.ones(input_shape)})

        if data_format == "NCHW":
            output = output.transpose(0, 3, 1, 2)

    return output


def export_cnn(filename: str, input_shape: List[int], data_format="NCHW"):
    input = tf.placeholder(tf.float32, shape=input_shape, name="input")

    filter = tf.constant(np.ones((5, 5, 1, 16)), dtype=tf.float32, name="weights")
    x = tf.nn.conv2d(input, filter, (1, 1, 1, 1), "SAME", data_format=data_format, name="conv2d")

    return export(x, filename)


def run_matmul(input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    b = tf.constant(np.ones((input_shape[1], 1)), dtype=tf.float32)

    x = tf.matmul(a, b)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: np.ones(input_shape)})

    return output


def export_matmul(filename: str, input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    b = tf.constant(np.ones((input_shape[1], 1)), dtype=tf.float32)

    x = tf.matmul(a, b)

    return export(x, filename)


def run_add(input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    b = tf.constant(np.ones((input_shape[1], 1)), dtype=tf.float32)

    x = tf.add(a, b)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: np.ones(input_shape)})

    return output


def export_add(filename: str, input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    b = tf.constant(np.ones((input_shape[0], 1)), dtype=tf.float32)

    x = tf.add(a, b)

    return export(x, filename)


def run_transpose(input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.transpose(a, perm=(0, 3, 1, 2))

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: np.ones(input_shape)})

    return output


def export_transpose(filename: str, input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.transpose(a, perm=(0, 3, 1, 2))

    return export(x, filename)


def run_reshape(input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")

    last_size = 1
    for i in input_shape[1:]:
        last_size *= i

    x = tf.reshape(a, [-1, last_size])

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: np.ones(input_shape)})

    return output


def export_reshape(filename: str, input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")

    last_size = 1
    for i in input_shape[1:]:
        last_size *= i

    x = tf.reshape(a, [-1, last_size])

    return export(x, filename)


def run_expand_dims(input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.expand_dims(a, axis=0)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: np.ones(input_shape)})

    return output


def export_expand_dims(filename: str, input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.expand_dims(a, axis=0)

    return export(x, filename)


def run_squeeze(input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.squeeze(a, axis=[0, 3])

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: np.ones(input_shape)})

    return output


def export_squeeze(filename: str, input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.squeeze(a, axis=[0, 3])

    return export(x, filename)


def run_sub(input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    b = tf.constant(np.ones((input_shape[0], 1)), dtype=tf.float32)

    x = tf.subtract(a, b)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: np.ones(input_shape)})

    return output


def export_sub(filename: str, input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    b = tf.constant(np.ones((input_shape[0], 1)), dtype=tf.float32)

    x = tf.subtract(a, b)

    return export(x, filename)


def run_mul(input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    b = tf.constant(np.array([1.0, 2.0, 3.0, 4.0]).reshape(input_shape), dtype=tf.float32)

    x = tf.multiply(a, b)

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: np.array([1.0, 2.0, 3.0, 4.0]).reshape(input_shape)})

    return output


def export_mul(filename: str, input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    b = tf.constant(np.array([1.0, 2.0, 3.0, 4.0]).reshape(input_shape), dtype=tf.float32)

    x = tf.multiply(a, b)

    return export(x, filename)


def export_strided_slice(filename: str, input_shape: List[int] = [3, 2, 3]):
    t = tf.placeholder(tf.float32, shape=input_shape, name="input")
    out = tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])

    return export(out, filename)


def run_strided_slice(input):
    t = tf.constant(input, dtype=tf.float32)
    out = tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])

    with tf.Session() as sess:
        output = sess.run(out)

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
