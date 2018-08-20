import unittest
import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

from tensorflow_encrypted.layers import AveragePooling2D


class TestAveragePooling2D(unittest.TestCase):
    def test_tiled_forward(self) -> None:
        batch_size, channels_in = 2, 2
        img_height, img_width = 8, 8
        input_shape = (batch_size, channels_in, img_height, img_width)
        n_elements = batch_size * channels_in * img_height * img_width
        input_pool = np.arange(n_elements, dtype=np.float32).reshape(input_shape)

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        # pooling in pond
        with tfe.protocol.Pond(*config.players) as prot:
            x_in = prot.define_public_variable(input_pool)
            pool = AveragePooling2D(pool_size=2)
            pool_out_pond = pool.forward(x_in)

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())
                out_pond = pool_out_pond.eval(sess)

        # reset tf graph
        tf.reset_default_graph()

        # pooling in tf
        with tf.Session() as sess:
            x = tf.Variable(input_pool, dtype=tf.float32)
            x_NHWC = tf.transpose(x, (0, 2, 3, 1))
            ksize = [1, 2, 2, 1]
            pool_out_tf = tf.nn.avg_pool(x_NHWC, ksize=ksize, strides=ksize, padding="SAME")
            out_tf = sess.run(pool_out_tf).transpose(0, 3, 1, 2)

        np.testing.assert_array_almost_equal(out_pond, out_tf, decimal=3)


    def test_backward(self):
        pass


if __name__ == '__main__':
    unittest.main()
