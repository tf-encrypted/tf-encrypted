import math
from typing import List, Union

import tensorflow as tf

from .tensor import AbstractTensor
from .prime import PrimeTensor


def binarize(tensor: AbstractTensor, prime: int=67) -> PrimeTensor:
    with tf.name_scope('binarize'):
        BITS = tensor.int_type.size * 8
        assert prime > BITS, prime

        final_shape = [1] * len(tensor.shape) + [BITS]
        bitwidths = tf.range(BITS, dtype=tensor.value.dtype)
        bitwidths = tf.reshape(bitwidths, final_shape)

        val = tf.expand_dims(tensor.value, -1)
        val = tf.bitwise.bitwise_and(tf.bitwise.right_shift(val, bitwidths), 1)

        return PrimeTensor.from_native(val, prime)


def im2col(tensor: Union[List[tf.Tensor], AbstractTensor], h_filter: int, w_filter: int,
           padding: str, strides: int) -> Union[List[tf.Tensor], tf.Tensor]:

    if isinstance(tensor, AbstractTensor):
        x = [tensor.value]
    else:
        x = tensor

    with tf.name_scope('im2col'):
        # we need NHWC because tf.extract_image_patches expects this
        NHWC_tensors = [tf.transpose(xi, [0, 2, 3, 1]) for xi in x]
        channels = int(NHWC_tensors[0].shape[3])
        # extract patches
        patch_tensors = [
            tf.extract_image_patches(
                xi,
                ksizes=[1, h_filter, w_filter, 1],
                strides=[1, strides, strides, 1],
                rates=[1, 1, 1, 1],
                padding=padding
            )
            for xi in NHWC_tensors
        ]

        # change back to NCHW
        patch_tensors_NCHW = [
            tf.reshape(
                tf.transpose(patches, [3, 1, 2, 0]),
                (h_filter, w_filter, channels, -1)
            )
            for patches in patch_tensors
        ]

        # reshape to x_col
        x_col_tensors = [
            tf.reshape(
                tf.transpose(x_col_NHWC, [2, 0, 1, 3]),
                (channels * h_filter * w_filter, -1)
            )
            for x_col_NHWC in patch_tensors_NCHW
        ]

        if isinstance(tensor, AbstractTensor):
            return type(tensor)(x_col_tensors[0])

        return x_col_tensors


def conv2d(x: AbstractTensor, y: AbstractTensor, strides: int, padding: str) -> AbstractTensor:
    assert isinstance(x, AbstractTensor), type(x)
    assert isinstance(y, AbstractTensor), type(y)

    h_filter, w_filter, d_filters, n_filters = map(int, y.shape)
    n_x, d_x, h_x, w_x = map(int, x.shape)
    if padding == "SAME":
        h_out = int(math.ceil(float(h_x) / float(strides)))
        w_out = int(math.ceil(float(w_x) / float(strides)))
    if padding == "VALID":
        h_out = int(math.ceil(float(h_x - h_filter + 1) / float(strides)))
        w_out = int(math.ceil(float(w_x - w_filter + 1) / float(strides)))

    X_col = im2col(x, h_filter, w_filter, padding, strides)
    W_col = y.transpose(perm=(3, 2, 0, 1)).reshape([int(n_filters), -1])
    out = W_col.dot(X_col)

    out = out.reshape([n_filters, h_out, w_out, n_x])
    out = out.transpose(perm=(3, 0, 1, 2))

    return out
