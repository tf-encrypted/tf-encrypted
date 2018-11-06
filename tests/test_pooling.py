import unittest
import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from typing import Tuple

from tf_encrypted.layers import AveragePooling2D


fixtureType = Tuple[np.ndarray, Tuple[int, int, int, int], tfe.LocalConfig]


class TestAveragePooling2D(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def _get_fixtures(self, even=True) -> fixtureType:
        if even:
            batch_size, channels_in = 2, 2
            img_height, img_width = 8, 8
        else:
            batch_size, channels_in = 1, 1
            img_height, img_width = 5, 11
        input_shape = (batch_size, channels_in, img_height, img_width)

        n_elements = batch_size * channels_in * img_height * img_width
        input_pool = np.ones(n_elements, dtype=np.float32).reshape(input_shape)

        return input_pool, input_shape

    def _tf_tiled_forward(self, input_pool: np.ndarray) -> np.ndarray:
        x = tf.constant(input_pool, dtype=tf.float32)
        x_NHWC = tf.transpose(x, (0, 2, 3, 1))
        ksize = [1, 2, 2, 1]
        pool_out_tf = tf.nn.avg_pool(
            x_NHWC, ksize=ksize, strides=ksize, padding="VALID", data_format='NHWC')

        with tf.Session() as sess:
            out_tf = sess.run(pool_out_tf).transpose(0, 3, 1, 2)

        return out_tf

    def _generic_tiled_forward(self, t_type: str, even: bool = True) -> None:
        assert t_type in ['public', 'private', 'masked']
        input_pool, input_shape = self._get_fixtures(even)

        # pooling in pond
        with tfe.protocol.Pond() as prot:
            if t_type == 'public':
                x_in = prot.define_public_variable(input_pool)
            elif t_type in ['private', 'masked']:
                x_in = prot.define_private_variable(input_pool)
            if t_type == 'masked':
                x_in = prot.mask(x_in)
            pool = AveragePooling2D(list(input_shape), pool_size=2, padding="VALID")
            pool_out_pond = pool.forward(x_in)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                if t_type in ['private', 'masked']:
                    out_pond = sess.run(pool_out_pond.reveal())
                else:
                    out_pond = sess.run(pool_out_pond)

        # reset tf graph
        tf.reset_default_graph()

        # pooling in tf
        out_tf = self._tf_tiled_forward(input_pool)

        np.testing.assert_array_almost_equal(out_pond, out_tf, decimal=3)

    def test_public_tiled_forward(self) -> None:
        self._generic_tiled_forward('public', True)

    def test_public_forward(self) -> None:
        self._generic_tiled_forward('public', False)

    def test_private_tiled_forward(self) -> None:
        self._generic_tiled_forward('private')

    def test_private_forward(self) -> None:
        self._generic_tiled_forward('private', False)

    def test_masked_tiled_forward(self) -> None:
        self._generic_tiled_forward('masked')

    def test_masked_forward(self) -> None:
        self._generic_tiled_forward('masked', False)


if __name__ == '__main__':
    unittest.main()
