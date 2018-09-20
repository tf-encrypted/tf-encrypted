import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, strides=1):
    # Zero-pad the input
    p = padding
    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    else:
        x_padded = x
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 strides)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

class TestIm2col(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_forward(self):
        batch_size, channels_in, channels_out = 4, 3, 16
        img_height, img_width = 10, 10
        h_filter, w_filter, strides = 2, 2, 2
        padding = 'SAME'
        input_shape = (batch_size, channels_in, img_height, img_width)
        input_im2col = np.random.normal(size=input_shape).astype(np.float32)

        config = tfe.LocalConfig(['server0', 'server1', 'crypto_producer'])

        # im2col pond
        with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:
            x = prot.define_private_variable(input_im2col)
            x_col = prot.im2col(x, h_filter, w_filter, padding, strides)

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())
                x_col_pond = x_col.reveal().eval(sess)

        # im2col numpy
        x_col_np = im2col_indices(input_im2col, h_filter, w_filter, padding=0, strides=strides)

        assert(np.isclose(x_col_pond, x_col_np, atol=0.001).all())


if __name__ == '__main__':
    unittest.main()
