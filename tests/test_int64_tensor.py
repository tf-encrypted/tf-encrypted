import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

from tf_encrypted.tensor import int64factory, fixed64


class TestInt64Tensor(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_pond(self) -> None:

        with tfe.protocol.Pond(
            tensor_factory=int64factory,
            fixedpoint_config=fixed64,
        ) as prot:

            x = prot.define_private_variable(np.array([2, 2]), apply_scaling=False)
            y = prot.define_public_variable(np.array([2, 2]), apply_scaling=False)

            z = x * y

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                out = sess.run(z.reveal())
                np.testing.assert_array_almost_equal(out, [4, 4], decimal=3)

    def test_binarize(self) -> None:
        x = int64factory.tensor(tf.constant([
            2**62 + 3,
            2**63 - 1,
            2**63 - 2,
            -3
        ], shape=[2, 2], dtype=tf.int64))

        y = x.bits()

        expected = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]).reshape([2, 2, 64])

        with tf.Session() as sess:
            actual = sess.run(y.to_native())

        np.testing.assert_array_equal(actual, expected)

    def test_random_binarize(self) -> None:
        input = np.random.uniform(low=2**63 + 1, high=2**63 - 1, size=2000).astype(np.int64).tolist()
        x = int64factory.tensor(tf.constant(input, dtype=tf.int64))

        y = x.bits()

        with tf.Session() as sess:
            actual = sess.run(y.to_native())

        j = 0
        for i in input:
            if i < 0:
                binary = bin(((1 << 64) - 1) & i)[2:][::-1]
            else:
                binary = bin(i)
                binary = binary[2:].zfill(64)[::-1]
            bin_list = np.array(list(binary)).astype(np.int64)
            np.testing.assert_equal(actual[j], bin_list)
            j += 1

# TODO this test fails for now until default tensorflow supports int64 matmul
# class TestConv2D(unittest.TestCase):
#     def setUp(self):
#         tf.reset_default_graph()
#
#     def test_forward(self) -> None:
#         # input
#         batch_size, channels_in, channels_out = 32, 3, 64
#         img_height, img_width = 28, 28
#         input_shape = (batch_size, channels_in, img_height, img_width)
#         input_conv = np.random.normal(size=input_shape).astype(np.int64)
#
#         # filters
#         h_filter, w_filter, strides = 2, 2, 2
#         filter_shape = (h_filter, w_filter, channels_in, channels_out)
#         filter_values = np.random.normal(size=filter_shape).astype(np.int64)
#
#         inp = Int64Tensor(input_conv)
#         out = inp.conv2d(Int64Tensor(filter_values), strides)
#         with tf.Session() as sess:
#             actual = sess.run(out.to_native())
#
#         # reset graph
#         tf.reset_default_graph()
#
#         # convolution tensorflow
#         with tf.Session() as sess:
#             # conv input
#             x = tf.Variable(input_conv, dtype=tf.float32)
#             x_NHWC = tf.transpose(x, (0, 2, 3, 1))
#
#             # convolution Tensorflow
#             filters_tf = tf.Variable(filter_values, dtype=tf.float32)
#
#             conv_out_tf = tf.nn.conv2d(x_NHWC, filters_tf, strides=[1, strides, strides, 1],
#                                        padding="SAME")
#
#             sess.run(tf.global_variables_initializer())
#             out_tensorflow = sess.run(conv_out_tf).transpose(0, 3, 1, 2)
#
#         np.testing.assert_array_almost_equal(actual, out_tensorflow, decimal=3)


if __name__ == '__main__':
    unittest.main()
