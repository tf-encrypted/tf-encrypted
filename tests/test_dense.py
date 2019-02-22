import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.layers import Dense


class TestDense(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_forward(self) -> None:

         # input
        input_shape = [4, 5]
        input = np.random.normal(size=input_shape)

        # variables
        weight_shape = [5, 2]
        weights_init = np.random.normal(size = weight_shape)
        bias_shape = [2]
        bias_init = np.random.normal(size = bias_shape)

        with tfe.protocol.Pond() as prot:
            
            dense_input = prot.define_private_variable(input)

            dense_layer = Dense(input_shape, out_features = 5)

            dense_layer.initialize(initial_weights=weights_init, 
                                    initial_bias=bias_init)

            dense_out_pond = dense_layer.forward(dense_input)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                out_pond = sess.run(dense_out_pond.reveal())

        # reset graph
        tf.reset_default_graph()

        with tf.Session() as sess:
            x = tf.Variable(input, dtype=tf.float32)
            weights_tf = tf.Variable(weights_init, dtype=tf.float32)
            bias_tf = tf.Variable(bias_init, dtype=tf.float32)

            out = tf.matmul(x, weights_tf) + bias_tf

            sess.run(tf.global_variables_initializer())
            out_tensorflow = sess.run(out)

        np.testing.assert_array_almost_equal(out_pond, out_tensorflow, decimal=2)

    
    def test_backward(self) -> None:

         # input
        input_shape = [1, 5]
        input = np.random.normal(size=input_shape)

        # variables
        weight_shape = [5, 1]
        weights_init = np.random.normal(size = weight_shape)
        bias_shape = [1]
        bias_init = np.random.normal(size = bias_shape)
        weights_second_layer = np.random.normal(size=[1, 1])

        with tfe.protocol.Pond() as prot:
            
            dense_input = prot.define_private_variable(input)
            w = prot.define_private_variable(weights_second_layer)
            

            dense_layer = Dense(input_shape, out_features = 1)

            dense_layer.initialize(initial_weights=weights_init, 
                                    initial_bias=bias_init)

            dense_out_pond = dense_layer.forward(dense_input)

            loss = dense_out_pond * w

            # backward
            d_out = w
            d_x, d_w = dense_layer.backward(d_out, learning_rate=1.0)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                d_w_pond = sess.run(d_w.reveal())
                d_x_pond = sess.run(d_x.reveal())
                

        # reset graph
        tf.reset_default_graph()

        with tf.Session() as sess:
            x = tf.Variable(input, dtype=tf.float32)
            weights_tf = tf.Variable(weights_init, dtype=tf.float32)
            bias_tf = tf.Variable(bias_init, dtype=tf.float32)

            out_tf = tf.matmul(x, weights_tf) + bias_tf

            # multiply conv output with some matrix
            w = tf.Variable(weights_second_layer, dtype=tf.float32)
            loss = out_tf * w

            # backward
            d_x, d_w = tf.gradients(xs=[x, weights_tf], ys=loss)

            sess.run(tf.global_variables_initializer())
            d_x_tf, d_w_tf = sess.run([d_x, d_w])

        np.testing.assert_array_almost_equal(d_w_pond, d_w_tf, decimal=2)
        np.testing.assert_array_almost_equal(d_x_pond, d_x_tf, decimal=2)


if __name__ == '__main__':
    unittest.main()
