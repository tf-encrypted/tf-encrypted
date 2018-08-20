import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

class TestConv2D(unittest.TestCase):

    def test_forward(self):
        # input
        batch_size, channels_in, channels_out = 32, 3, 64
        img_height, img_width = 28, 28
        input_shape = (batch_size, channels_in, img_height, img_width)
        input_conv = np.random.normal(size=input_shape).astype(np.float32)

        # filters
        h_filter, w_filter, strides, padding = 2, 2, 2, 0
        filter_shape = (h_filter, w_filter, channels_in, channels_out)
        filter_values = np.random.normal(size=filter_shape)

        config = tfe.LocalConfig(3)

        # convolution pond
        with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:
        
            conv_input = prot.define_private_variable(input_conv)
            conv_layer = tfe.layer.Conv2D(filter_shape, strides=2)
            conv_layer.initialize(input_shape, initial_weights=filter_values)
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

    def test_backward(self):
        pass


if __name__ == '__main__':
    unittest.main()
