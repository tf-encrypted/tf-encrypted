import unittest

import numpy as np
import tensorflow as tf
from tf_encrypted.tensor import int32factory


class TestInt32Tensor(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_binarize(self) -> None:
        x = int32factory.tensor(np.array([
            2**32 + 3,  # == 3
            2**31 - 1,  # max
            2**31,  # min
            -3
        ]).reshape(2, 2))

        y = x.bits()

        expected = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]).reshape([2, 2, 32])

        with tf.Session() as sess:
            actual = sess.run(y.to_native())

        np.testing.assert_array_equal(actual, expected)

    def test_random_binarize(self) -> None:
        input = np.random.uniform(low=2**31 + 1, high=2**31 - 1, size=2000).astype('int32')
        x = int32factory.tensor(input)

        y = x.bits()

        with tf.Session() as sess:
            actual = sess.run(y.to_native())

        j = 0
        for i in input.tolist():
            if i < 0:
                binary = bin(((1 << 32) - 1) & i)[2:][::-1]
            else:
                binary = bin(i)
                binary = binary[2:].zfill(32)[::-1]
            bin_list = np.array(list(binary)).astype(np.int32)
            np.testing.assert_equal(actual[j], bin_list)
            j += 1


class TestConv2D(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_forward(self) -> None:
        # input
        batch_size, channels_in, channels_out = 32, 3, 64
        img_height, img_width = 28, 28
        input_shape = (batch_size, channels_in, img_height, img_width)
        input_conv = np.random.normal(size=input_shape).astype(np.int32)

        # filters
        h_filter, w_filter, strides = 2, 2, 2
        filter_shape = (h_filter, w_filter, channels_in, channels_out)
        filter_values = np.random.normal(size=filter_shape).astype(np.int32)

        inp = int32factory.tensor(input_conv)
        out = inp.conv2d(int32factory.tensor(filter_values), strides)
        with tf.Session() as sess:
            actual = sess.run(out.to_native())

        # reset graph
        tf.reset_default_graph()

        # convolution tensorflow
        with tf.Session() as sess:
            # conv input
            x = tf.Variable(input_conv, dtype=tf.float32)
            x_NHWC = tf.transpose(x, (0, 2, 3, 1))

            # convolution Tensorflow
            filters_tf = tf.Variable(filter_values, dtype=tf.float32)

            conv_out_tf = tf.nn.conv2d(x_NHWC, filters_tf, strides=[1, strides, strides, 1],
                                       padding="SAME")

            sess.run(tf.global_variables_initializer())
            out_tensorflow = sess.run(conv_out_tf).transpose(0, 3, 1, 2)

        np.testing.assert_array_almost_equal(actual, out_tensorflow, decimal=3)


if __name__ == '__main__':
    unittest.main()
