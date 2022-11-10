# pylint: disable=missing-docstring
import math
import random
import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.tensor import int100factory


class TestInt100Tensor(unittest.TestCase):
    def core_test_binarize(
        self,
        raw,
        shape,
        modulus,
        bitlen,
        ensure_positive_interpretation,
    ) -> None:
        def as_bits(x: int, min_bitlength):
            bits = [int(b) for b in "{0:b}".format(x)]
            bits = [0] * (min_bitlength - len(bits)) + bits
            return list(reversed(bits))

        expected = np.array([as_bits((modulus + x) % modulus, bitlen) for x in raw])
        expected = expected.reshape(shape + (bitlen,))

        x = int100factory.tensor(np.array(raw).reshape(shape))
        epi = ensure_positive_interpretation
        actual = x.bits(ensure_positive_interpretation=epi).to_native()

        np.testing.assert_array_equal(actual, expected)

    def test_binarize_positive(self) -> None:
        lower = -int100factory.modulus // 2 + 1
        upper = int100factory.modulus // 2

        random.seed(1234)
        raw = [-1, 0, 1] + [random.randint(lower, upper) for _ in range(256 - 3)]
        shape = (2, 2, 2, -1)

        bitlen = math.ceil(math.log2(int100factory.modulus))
        modulus = int100factory.modulus
        self.core_test_binarize(raw, shape, modulus, bitlen, True)

    def test_binarize_symmetric(self) -> None:
        lower = -int100factory.modulus // 2 + 1
        upper = int100factory.modulus // 2

        random.seed(1234)
        raw = [-1, 0, 1] + [random.randint(lower, upper) for _ in range(256 - 3)]
        shape = (2, 2, 2, -1)

        bitlen = math.ceil(math.log2(int100factory.modulus))
        modulus = 2**bitlen
        self.core_test_binarize(raw, shape, modulus, bitlen, False)


class TestConv2D(unittest.TestCase):
    def test_forward(self) -> None:

        # input
        batch_size, channels_in, channels_out = 32, 3, 64
        img_height, img_width = 28, 28
        input_shape = (batch_size, channels_in, img_height, img_width)
        input_conv = np.random.normal(size=input_shape).astype(np.int32)

        # filters
        h_filter, w_filter = 2, 2
        strides = [2, 2]
        filter_shape = (h_filter, w_filter, channels_in, channels_out)
        filter_values = np.random.normal(size=filter_shape).astype(np.int32)

        inp = int100factory.tensor(input_conv)
        out = inp.conv2d(int100factory.tensor(filter_values), strides)
        actual = out.to_native()

        # conv input
        x = tf.Variable(input_conv, dtype=tf.float32)
        x_nhwc = tf.transpose(x, (0, 2, 3, 1))

        # conv filter
        filters_tf = tf.Variable(filter_values, dtype=tf.float32)

        conv_out_tf = tf.nn.conv2d(
            x_nhwc, filters_tf, strides=[1, strides[0], strides[1], 1], padding="SAME"
        )
        expected = tf.transpose(conv_out_tf, perm=[0, 3, 1, 2])
        np.testing.assert_array_almost_equal(actual, expected, decimal=3)


if __name__ == "__main__":
    unittest.main()
