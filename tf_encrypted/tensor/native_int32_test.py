# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.tensor import int32factory


class TestInt32Tensor(unittest.TestCase):
    def test_binarize(self) -> None:
        x = int32factory.tensor(
            np.array(
                [2**32 + 3, 2**31 - 1, 2**31, -3]  # == 3  # max  # min
            ).reshape(2, 2)
        )

        y = x.bits()

        # fmt: off
        expected = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]).reshape([2, 2, 32])
        # fmt: on

        np.testing.assert_array_equal(y.to_native(), expected)

    def test_random_binarize(self) -> None:
        x_in = np.random.uniform(
            low=2**31 + 1,
            high=2**31 - 1,
            size=2000,
        ).astype("int32")
        x = int32factory.tensor(x_in)
        y = x.bits()
        actual = y.to_native()

        j = 0
        for i in x_in.tolist():
            if i < 0:
                binary = bin(((1 << 32) - 1) & i)[2:][::-1]
            else:
                binary = bin(i)
                binary = binary[2:].zfill(32)[::-1]
            bin_list = np.array(list(binary)).astype(np.int32)
            np.testing.assert_equal(actual[j], bin_list)
            j += 1


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

        inp = int32factory.tensor(input_conv)
        out = inp.conv2d(int32factory.tensor(filter_values), strides)
        actual = out.to_native()

        # conv input
        x = tf.Variable(input_conv, dtype=tf.int32)
        x_nhwc = tf.transpose(x, (0, 2, 3, 1))

        # conv filter
        filters_tf = tf.Variable(filter_values, dtype=tf.int32)

        conv_out_tf = tf.nn.conv2d(
            x_nhwc,
            filters_tf,
            strides=[1, strides[0], strides[1], 1],
            padding="SAME",
        )

        out_tensorflow = tf.transpose(conv_out_tf, perm=[0, 3, 1, 2])
        np.testing.assert_array_equal(actual, out_tensorflow)


if __name__ == "__main__":
    unittest.main()
