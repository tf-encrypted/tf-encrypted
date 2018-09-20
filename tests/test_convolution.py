import unittest
import itertools

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

np.random.seed(1)

class TestConv2D(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_forward(self) -> None:
        # input
        batch_size, channels_in, channels_out = 4, 3, 16
        img_height, img_width = 10, 10
        input_shape = (batch_size, channels_in, img_height, img_width)
        input_conv = np.random.normal(size=input_shape).astype(np.float32)

        # filters
        h_filter, w_filter, strides = 2, 2, 2
        filter_shape = (h_filter, w_filter, channels_in, channels_out)
        filter_values = np.random.normal(size=filter_shape)

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        # convolution pond
        with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:

            conv_input = prot.define_private_variable(input_conv)
            conv_layer = tfe.layers.Conv2D(input_shape, filter_shape, strides=2)
            conv_layer.initialize(initial_weights=filter_values)
            conv_out_pond = conv_layer.forward(conv_input)

            with config.session() as sess:

                sess.run(tf.global_variables_initializer())
                # outputs
                out_pond = conv_out_pond.reveal().eval(sess)

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

        np.testing.assert_array_almost_equal(out_pond, out_tensorflow, decimal=3)


    def test_backward(self) -> None:
        batch_size, channels_in, channels_out = 8, 3, 4
        img_height, img_width = 8, 8
        input_shape = (batch_size, channels_in, img_height, img_width)
        input_conv = np.random.normal(size=input_shape).astype(np.float32)

        # filters
        h_filter, w_filter, strides = 2, 2, 2
        filter_shape = (h_filter, w_filter, channels_in, channels_out)
        filter_values = np.random.normal(size=filter_shape)

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        # convolution pond
        with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:
            # forward
            conv_input = prot.define_private_variable(input_conv)
            conv_layer = tfe.layers.Conv2D(input_shape, filter_shape, strides=2)
            conv_layer.initialize(initial_weights=filter_values)
            conv_out = conv_layer.forward(conv_input)

            s = tuple(map(int, conv_out.shape))
            weights_second_layer = np.random.normal(size=s)
            w = prot.define_private_variable(weights_second_layer)
            loss = conv_out * w

            # backward
            d_out = w
            d_x, d_w = conv_layer.backward(d_out, learning_rate=1.0)

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())
                # outputs
                # d_x_pond = d_x.reveal().eval(sess)
                d_w_pond = d_w.reveal().eval(sess)

        # reset graph
        tf.reset_default_graph()

        # convolution tensorflow
        with tf.Session() as sess:
            x = tf.Variable(input_conv, dtype=tf.float32)
            x_nhwc = tf.transpose(x, (0, 2, 3, 1))
            filters_tf = tf.Variable(filter_values, dtype=tf.float32)

            # forward
            conv_out = tf.nn.conv2d(x_nhwc, filters_tf, strides=[1, strides, strides, 1],
                                       padding="SAME")
            conv_out_nchw = tf.transpose(conv_out, (0, 3, 1, 2))

            # multiply conv output with some matrix
            w = tf.Variable(weights_second_layer, dtype=tf.float32)
            loss = conv_out_nchw * w

            # backward
            d_x, d_w, d_conv = tf.gradients(xs=[x, filters_tf, conv_out_nchw], ys=loss)

            sess.run(tf.global_variables_initializer())
            d_x_tensorflow, d_w_tensorflow = sess.run([d_x, d_w])

        # match derivative of weights
        np.testing.assert_array_almost_equal(d_w_tensorflow, d_w_pond, decimal=2)

if __name__ == '__main__':
    unittest.main()
