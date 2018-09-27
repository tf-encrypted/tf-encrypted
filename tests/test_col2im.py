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


def col2im_indices(cols, x_shape, field_height, field_width, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded))
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class TestCol2im(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_col2im(self):
        batch_size, channels_in = 4, 3
        img_height, img_width = 10, 10
        h_filter, w_filter, strides = 2, 2, 2
        padding = 'SAME'
        patch_size = 12
        n_patches = 100
        col_shape = (patch_size, n_patches)
        x_im_shape = (batch_size, channels_in, img_height, img_width)
        input_col2im = np.random.normal(size=col_shape).astype(np.float32)

        config = tfe.LocalConfig(['server0', 'server1', 'crypto_producer'])

        # col2im pond
        with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:
            x_col = prot.define_private_variable(input_col2im)
            x_im = prot.col2im(x_col, x_im_shape, h_filter, w_filter, padding, strides)

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())
                x_im_pond = x_im.reveal().eval(sess)

        # col2im numpy
        x_im_np = col2im_indices(input_col2im, x_im_shape, h_filter, w_filter, padding=0, stride=strides)

        assert(np.isclose(x_im_pond, x_im_np, atol=0.001).all())


if __name__ == '__main__':
    unittest.main()
