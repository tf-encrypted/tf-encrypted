import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe


class TestRelu(unittest.TestCase):
    def test_forward(self):
        input_relu = np.array([-1.0, -0.5, 0.5, 3.0]).astype(np.float32)

        config = tfe.LocalConfig(3)

        # relu pond
        with tfe.protocol.Pond(*config.players) as prot:

            relu_input = prot.define_private_variable(input_relu)
            relu_layer = tfe.layer.Relu()

            relu_out_pond = relu_layer.forward(relu_input)

            with config.session() as sess:

                sess.run(tf.global_variables_initializer())
                # outputs
                out_pond = relu_out_pond.reveal().eval(sess)

            # reset graph
            tf.reset_default_graph()

            with tf.Session() as sess:
                x = tf.Variable(input_relu, dtype=tf.float32)

                relu_out_tf = tf.nn.relu(x)

                sess.run(tf.global_variables_initializer())

                out_tensorflow = sess.run(relu_out_tf)
                print(out_tensorflow)

        assert(np.isclose(out_pond, out_tensorflow, atol=0.6).all())


if __name__ == '__main__':
    unittest.main()
