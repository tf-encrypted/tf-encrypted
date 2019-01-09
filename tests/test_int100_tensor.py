import unittest
import math
import random

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

from tf_encrypted.tensor import int100factory, fixed100_ni


class TestInt100Tensor(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def test_pond(self) -> None:

        with tfe.protocol.Pond(
            None,
            tensor_factory=int100factory,
            fixedpoint_config=fixed100_ni,
        ) as prot:

            x = prot.define_private_variable(np.array([2, 2]), apply_scaling=False)
            y = prot.define_public_variable(np.array([2, 2]), apply_scaling=False)

            z = x * y

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                actual = sess.run(z.reveal())

            expected = np.array([4, 4])
            np.testing.assert_array_almost_equal(actual, expected, decimal=3)

    def core_test_binarize(self, raw, shape, modulus, bitlen, ensure_positive_interpretation) -> None:

        def as_bits(x: int, min_bitlength):
            bits = [int(b) for b in '{0:b}'.format(x)]
            bits = [0] * (min_bitlength - len(bits)) + bits
            return list(reversed(bits))

        expected = np.array([
            as_bits((modulus + x) % modulus, bitlen)
            for x in raw
        ]).reshape(shape + (bitlen,))

        x = int100factory.tensor(np.array(raw).reshape(shape))
        y = x.bits(ensure_positive_interpretation=ensure_positive_interpretation).to_native()

        with tf.Session() as sess:
            actual = sess.run(y)

        np.testing.assert_array_equal(actual, expected)

    def test_binarize_positive(self) -> None:
        lower = -int100factory.modulus // 2 + 1
        upper = int100factory.modulus // 2

        random.seed(1234)
        raw = [-1, 0, 1] + [
            random.randint(lower, upper)
            for _ in range(256 - 3)
        ]
        shape = (2, 2, 2, -1)

        bitlen = math.ceil(math.log2(int100factory.modulus))
        modulus = int100factory.modulus
        self.core_test_binarize(raw, shape, modulus, bitlen, True)

    def test_binarize_symmetric(self) -> None:
        lower = -int100factory.modulus // 2 + 1
        upper = int100factory.modulus // 2

        random.seed(1234)
        raw = [-1, 0, 1] + [
            random.randint(lower, upper)
            for _ in range(256 - 3)
        ]
        shape = (2, 2, 2, -1)

        bitlen = math.ceil(math.log2(int100factory.modulus))
        modulus = 2**bitlen
        self.core_test_binarize(raw, shape, modulus, bitlen, False)


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

        inp = int100factory.tensor(input_conv)
        out = inp.conv2d(int100factory.tensor(filter_values), strides)
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

            conv_out_tf = tf.nn.conv2d(
                x_NHWC,
                filters_tf,
                strides=[1, strides, strides, 1],
                padding="SAME"
            )

            sess.run(tf.global_variables_initializer())
            expected = sess.run(conv_out_tf).transpose(0, 3, 1, 2)

        np.testing.assert_array_almost_equal(actual, expected, decimal=3)


if __name__ == '__main__':
    unittest.main()
