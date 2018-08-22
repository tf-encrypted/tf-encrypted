import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

# TODO: Merge this with test_pooling.py

class TestAvgPooling(unittest.TestCase):

    def test_forward(self) -> None:

        # input
        input = np.array((1,0,1,0,
                          0,1,0,1,
                          1,0,1,0,
                          0,1,0,1,
                          1,0,1,0,
                          0,1,0,1)).reshape(1, 1, 4, 6)

        config = tfe.LocalConfig([
            'server_0',
            'server_1',
            'crypto_producer'
        ])

        # convolution pond
        with tfe.protocol.Pond(*config.players) as prot:

            pool_input = prot.define_public_variable(input)
            pool_layer = tfe.layers.AveragePooling2D([1, 1, 4, 6],
                                                     pool_size=(2, 3),
                                                     strides=(2, 3))
            pool_layer.initialize(input_shape=(1, 1, 4, 6))
            pool_out_pond = pool_layer.forward(pool_input)

            with config.session() as sess:

                sess.run(tf.global_variables_initializer())
                # outputs
                out_pond = pool_out_pond.eval(sess)
                print(f"result! {out_pond.shape} ... {out_pond}")

        # reset graph
        tf.reset_default_graph()

        # input
        input = np.array((1,0,1,0,
                          0,1,0,1,
                          1,0,1,0,
                          0,1,0,1,
                          1,0,1,0,
                          0,1,0,1)).reshape(1, 4, 6, 1)
        filter_shape = (1, 2, 3, 1)

        # pooling tensorflow
        with tf.Session() as sess:
            # conv input
            x = tf.Variable(input, dtype=tf.float32)

            stride = 2
            window = 2
            conv_out_tf = tf.nn.avg_pool(x,
                                         ksize=[1, 2, 3, 1],
                                         strides=[1, 2, 3, 1],
                                         padding="SAME")

            sess.run(tf.global_variables_initializer())
            out_tensorflow = sess.run(conv_out_tf).transpose(0, 3, 1, 2)
            print(f'woah! {out_tensorflow}')


        # np.testing.assert_array_almost_equal(out_pond, out_tensorflow, decimal=3)

    def test_backward(self):
        pass


if __name__ == '__main__':
    unittest.main()
