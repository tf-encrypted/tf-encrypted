import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.layers.activation import Tanh


class TestTanh(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_forward(self):
        input_shape = [4]
        input_tanh = np.array([-1.0, -0.5, 0.5, 3.0]).astype(np.float32)

        # tanh pond
        with tfe.protocol.Pond() as prot:

            tanh_input = prot.define_private_variable(input_tanh)
            tanh_layer = Tanh(input_shape)

            tanh_out_pond = tanh_layer.forward(tanh_input)

            with tfe.Session() as sess:

                sess.run(tf.global_variables_initializer())
                # outputs
                out_pond = sess.run(tanh_out_pond.reveal())

            # reset graph
            tf.reset_default_graph()

            with tf.Session() as sess:
                x = tf.Variable(input_tanh, dtype=tf.float32)

                tanh_out_tf = tf.nn.tanh(x)

                sess.run(tf.global_variables_initializer())

                out_tensorflow = sess.run(tanh_out_tf)

        assert(np.isclose(out_pond, out_tensorflow, atol=0.2).all())


if __name__ == '__main__':
    unittest.main()
