import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.tensor.int100 import Int100Tensor, Int100Factory


class TestInt100Tensor(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def test_pond(self) -> None:
        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        with tfe.protocol.Pond(
            *config.get_players('server0, server1, crypto_producer'),
            tensor_factory=Int100Factory(),
            use_noninteractive_truncation=True,
            verify_precision=True
        ) as prot:

            x = prot.define_private_variable(np.array([2, 2]), apply_scaling=False)
            y = prot.define_public_variable(np.array([2, 2]), apply_scaling=False)

            z = x * y

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())
                actual = z.reveal().eval(sess)

            expected = np.array([4, 4])
            np.testing.assert_array_almost_equal(actual, expected, decimal=3)

    def test_binarize(self) -> None:

        def as_bits(x: int, min_bitlength):
            bits = [int(b) for b in '{0:b}'.format(x)]
            bits = [0] * (min_bitlength - len(bits)) + bits
            return list(reversed(bits))

        x = Int100Tensor(np.array([
            0,
            -1,
            123456789,
            -123456789,
        ]).reshape([2, 2]))

        with tf.Session() as sess:
            actual = sess.run(
                x.convert_to_tensor().to_bits().to_native()
            )

        expected = np.array([
            as_bits((2**103 + (0)) % 2**103, 103),  # == as_bits(0, 103)
            as_bits((2**103 + (-1)) % 2**103, 103),
            as_bits((2**103 + (123456789)) % 2**103, 103),  # == as_bits(123456789, 103)
            as_bits((2**103 + (-123456789)) % 2**103, 103),
        ]).reshape([2, 2, 103])

        np.testing.assert_array_equal(actual, expected)


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

        inp = Int100Tensor(input_conv)
        out = inp.conv2d(Int100Tensor(filter_values), strides)
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
