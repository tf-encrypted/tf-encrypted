import math
from typing import Union, Optional

import tensorflow as tf
import numpy as np

from .factory import AbstractTensor


def binarize(tensor: tf.Tensor, bitsize: Optional[int] = None) -> tf.Tensor:
    """ Extract bits of values in `tensor`, returning a `tf.Tensor` with same dtype. """

    with tf.name_scope('binarize'):
        bitsize = bitsize or (tensor.dtype.size * 8)

        bit_indices_shape = [1] * len(tensor.shape) + [bitsize]
        bit_indices = tf.range(bitsize, dtype=tensor.dtype)
        bit_indices = tf.reshape(bit_indices, bit_indices_shape)

        val = tf.expand_dims(tensor, -1)
        val = tf.bitwise.bitwise_and(tf.bitwise.right_shift(val, bit_indices), 1)

        assert val.dtype == tensor.dtype
        return val


def bits(tensor: tf.Tensor, bitsize: Optional[int] = None) -> list:
    """ Extract bits of values in `tensor`, returning a list of tensors """

    with tf.name_scope('bits'):
        bitsize = bitsize or (tensor.dtype.size * 8)
        bits = [
            tf.bitwise.bitwise_and(tf.bitwise.right_shift(tensor, i), 1)
            for i in range(bitsize)
        ]
        return bits
        # return tf.stack(bits, axis=-1)


def im2col(
    x: Union[tf.Tensor, np.ndarray],
    h_filter: int,
    w_filter: int,
    padding: str,
    strides: int
) -> tf.Tensor:

    with tf.name_scope('im2col'):

        # we need NHWC because tf.extract_image_patches expects this
        NHWC_tensor = tf.transpose(x, [0, 2, 3, 1])
        channels = int(NHWC_tensor.shape[3])

        # extract patches
        patch_tensor = tf.extract_image_patches(
            NHWC_tensor,
            ksizes=[1, h_filter, w_filter, 1],
            strides=[1, strides, strides, 1],
            rates=[1, 1, 1, 1],
            padding=padding
        )

        # change back to NCHW
        patch_tensor_NCHW = tf.reshape(
            tf.transpose(patch_tensor, [3, 1, 2, 0]),
            (h_filter, w_filter, channels, -1)
        )

        # reshape to x_col
        x_col_tensor = tf.reshape(
            tf.transpose(patch_tensor_NCHW, [2, 0, 1, 3]),
            (channels * h_filter * w_filter, -1)
        )

        return x_col_tensor


def conv2d(
    x: AbstractTensor,
    y: AbstractTensor,
    strides: int,
    padding: str
) -> AbstractTensor:

    with tf.name_scope('conv2d'):

        h_filter, w_filter, d_filters, n_filters = map(int, y.shape)
        n_x, d_x, h_x, w_x = map(int, x.shape)
        if padding == 'SAME':
            h_out = int(math.ceil(float(h_x) / float(strides)))
            w_out = int(math.ceil(float(w_x) / float(strides)))
        elif padding == 'VALID':
            h_out = int(math.ceil(float(h_x - h_filter + 1) / float(strides)))
            w_out = int(math.ceil(float(w_x - w_filter + 1) / float(strides)))
        else:
            raise ValueError("Don't know padding method '{}'".format(padding))

        X_col = x.im2col(h_filter, w_filter, padding, strides)
        W_col = y.transpose([3, 2, 0, 1]).reshape([int(n_filters), -1])
        out = W_col.matmul(X_col)

        out = out.reshape([n_filters, h_out, w_out, n_x])
        out = out.transpose([3, 0, 1, 2])

        return out
