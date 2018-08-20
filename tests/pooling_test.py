import unittest
import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

from tensorflow_encrypted.layers import AveragePooling2D

np.loadtxt('fixtures/pooling.csv', separator=',')

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
        with tfe.protocol.Pond(*config.players) as prot:
            x_in = prot.define_public_variable(input_pool)
            pool = AveragePooling2D(pool_size=2)
            pool_out = pool.forward(x_in)


    def test_backward(self):
        pass
