import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

from tf_encrypted.layers import Batchnorm


class TestBatchnorm(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_forward(self) -> None:

        batch_size, channels_in, img_height, img_width = (32, 3, 28, 28)

        input_shape = [batch_size, channels_in, img_height, img_width]
        input_batchnorm = np.random.normal(size=input_shape).astype(np.float32)

        # I reshaped the input because tf.nn.batch_normalization doesn't reshape it
        # automatically However tf encrypted will reshape automatically the input
        mean = np.array([2.0, 1.5, 20.8]).reshape(1, channels_in, 1, 1).astype(np.float32)
        variance = np.array([0.5, 0.3, 0.1]).reshape(1, channels_in, 1, 1).astype(np.float32)
        scale = np.array([0.3, 0.5, 0.8]).reshape(1, channels_in, 1, 1).astype(np.float32)
        offset = np.array([1.5, 1.2, 1.4]).reshape(1, channels_in, 1, 1).astype(np.float32)
        variance_epsilon = 1e-8

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:
            batchnorm_input = prot.define_private_variable(input_batchnorm)

            batchnorm_layer = Batchnorm(input_shape, mean, variance, scale, offset)
            batchnorm_layer.initialize()
            batchnorm_out_pond = batchnorm_layer.forward(batchnorm_input)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                out_pond = sess.run(batchnorm_out_pond.reveal())

            # reset graph
            tf.reset_default_graph()

            with tf.Session() as sess:
                x = tf.Variable(input_batchnorm, dtype=tf.float32)

                batchnorm_out_tf = tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon)

                sess.run(tf.global_variables_initializer())

                out_tensorflow = sess.run(batchnorm_out_tf)

                np.testing.assert_array_almost_equal(out_pond, out_tensorflow, decimal=1)


if __name__ == '__main__':
    unittest.main()
