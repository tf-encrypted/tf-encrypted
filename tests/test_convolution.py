import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestConv2D(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_forward(self) -> None:
        # input
        batch_size, channels_in, channels_out = 32, 3, 64
        img_height, img_width = 28, 28
        input_shape = (batch_size, channels_in, img_height, img_width)
        input_conv = np.random.normal(size=input_shape).astype(np.float32)

        # filters
        h_filter, w_filter, strides = 2, 2, 2
        filter_shape = (h_filter, w_filter, channels_in, channels_out)
        filter_values = np.random.normal(size=filter_shape)

        # convolution pond
        with tfe.protocol.Pond() as prot:

            conv_input = prot.define_private_variable(input_conv)
            conv_layer = tfe.layers.Conv2D(input_shape, filter_shape, strides=2)
            conv_layer.initialize(initial_weights=filter_values)
            conv_out_pond = conv_layer.forward(conv_input)

            with tfe.Session() as sess:

                sess.run(tf.global_variables_initializer())
                # outputs
                out_pond = sess.run(conv_out_pond.reveal())

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

        np.testing.assert_allclose(out_pond, out_tensorflow, atol=0.01)

    def test_forward_bias(self) -> None:
        # input
        batch_size, channels_in, channels_out = 32, 3, 64
        img_height, img_width = 28, 28
        input_shape = (batch_size, channels_in, img_height, img_width)
        input_conv = np.random.normal(size=input_shape).astype(np.float32)

        # filters
        h_filter, w_filter, strides = 2, 2, 2
        filter_shape = (h_filter, w_filter, channels_in, channels_out)
        filter_values = np.random.normal(size=filter_shape)

        # convolution pond
        with tfe.protocol.Pond() as prot:

            conv_input = prot.define_private_variable(input_conv)
            conv_layer = tfe.layers.Conv2D(input_shape, filter_shape, strides=2)

            output_shape = conv_layer.get_output_shape()

            bias = np.random.uniform(size=output_shape[1:])

            conv_layer.initialize(initial_weights=filter_values, initial_bias=bias)
            conv_out_pond = conv_layer.forward(conv_input)

            with tfe.Session() as sess:

                sess.run(tf.global_variables_initializer())
                # outputs
                out_pond = sess.run(conv_out_pond.reveal())

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
            out_tensorflow += bias

        np.testing.assert_allclose(out_pond, out_tensorflow, atol=0.01)

    def test_backward(self):
        pass


if __name__ == '__main__':
    unittest.main()
