# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.tensor import int64factory


class TestInt64Tensor(unittest.TestCase):
    def test_binarize(self) -> None:
        x = int64factory.tensor(
            tf.constant(
                [2**62 + 3, 2**63 - 1, 2**63 - 2, -3],
                shape=[2, 2],
                dtype=tf.int64,
            )
        )

        y = x.bits()

        # fmt: off
        expected = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]).reshape([2, 2, 64])
        # fmt: on

        actual = y.to_native()
        np.testing.assert_array_equal(actual, expected)

    def test_random_binarize(self) -> None:
        x_in = (
            np.random.uniform(
                low=2**63 + 1,
                high=2**63 - 1,
                size=2000,
            )
            .astype(np.int64)
            .tolist()
        )
        x = int64factory.tensor(tf.constant(x_in, dtype=tf.int64))

        y = x.bits()
        actual = y.to_native()

        j = 0
        for i in x_in:
            if i < 0:
                binary = bin(((1 << 64) - 1) & i)[2:][::-1]
            else:
                binary = bin(i)
                binary = binary[2:].zfill(64)[::-1]
            bin_list = np.array(list(binary)).astype(np.int64)
            np.testing.assert_equal(actual[j], bin_list)
            j += 1


class TestConv2D(unittest.TestCase):
    def test_forward(self) -> None:
        # input
        batch_size, channels_in, channels_out = 32, 3, 64
        img_height, img_width = 28, 28
        input_shape = (batch_size, channels_in, img_height, img_width)
        input_conv = np.random.normal(size=input_shape).astype(np.int64)

        # filters
        h_filter, w_filter = 2, 2
        strides = [2, 2]
        filter_shape = (h_filter, w_filter, channels_in, channels_out)
        filter_values = np.random.normal(size=filter_shape).astype(np.int64)

        x_in = int64factory.tensor(input_conv)
        out = x_in.conv2d(int64factory.tensor(filter_values), strides)
        actual = out.to_native()

        # conv input
        x = tf.Variable(input_conv, dtype=tf.float32)
        x_nhwc = tf.transpose(x, (0, 2, 3, 1))

        # conv filter
        filters_tf = tf.Variable(filter_values, dtype=tf.float32)

        conv_out_tf = tf.nn.conv2d(
            x_nhwc,
            filters_tf,
            strides=[1, strides[0], strides[1], 1],
            padding="SAME",
        )

        out_tensorflow = tf.transpose(conv_out_tf, perm=[0, 3, 1, 2])
        np.testing.assert_array_almost_equal(actual, out_tensorflow, decimal=3)


if __name__ == "__main__":
    unittest.main()
