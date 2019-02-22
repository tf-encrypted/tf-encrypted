import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.layers import Dense


class TestDense(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_dense(self) -> None:

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


if __name__ == '__main__':
    unittest.main()
