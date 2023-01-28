"""Commonly used tensor functions."""
import math
from typing import Optional
from typing import Union

import numpy as np
import tensorflow as tf


def binarize(
    tensor: tf.Tensor,
    bitsize: Optional[int] = None,
) -> tf.Tensor:
    """Extract bits of values in `tensor`, returning a `tf.Tensor` with same
    dtype."""

    with tf.name_scope("binarize"):
        bitsize = bitsize or (tensor.dtype.size * 8)

        bit_indices_shape = [1] * len(tensor.shape) + [bitsize]
        bit_indices = tf.range(bitsize, dtype=tensor.dtype)
        bit_indices = tf.reshape(bit_indices, bit_indices_shape)

        val = tf.expand_dims(tensor, -1)
        val = tf.bitwise.bitwise_and(tf.bitwise.right_shift(val, bit_indices), 1)

        assert val.dtype == tensor.dtype
        return val


def bits(
    tensor: tf.Tensor,
    bitsize: Optional[int] = None,
) -> list:
    """Extract bits of values in `tensor`, returning a list of tensors."""

    with tf.name_scope("bits"):
        bitsize = bitsize or (tensor.dtype.size * 8)
        the_bits = [
            tf.bitwise.bitwise_and(tf.bitwise.right_shift(tensor, i), 1)
            for i in range(bitsize)
        ]
        return the_bits
        # return tf.stack(bits, axis=-1)


def im2patches(x, patch_size, strides=[1, 1], padding="SAME", data_format="NCHW"):
    """
    :param x: a 4-D Tensor.
    """

    with tf.name_scope("im2patches"):
        # To NHWC
        if data_format == "NCHW":
            x = tf.transpose(x, [0, 2, 3, 1])

        # we need NHWC because tf.extract_image_patches expects this
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, patch_size[0], patch_size[1], 1],
            strides=[1, strides[0], strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=padding,
        )
        # To NCHW
        if data_format == "NCHW":
            patches = tf.transpose(patches, [0, 3, 1, 2])

    return patches

def patches2im(
    patches,
    patch_size,
    strides=[1, 1],
    padding="SAME",
    img_size=None,
    consolidation="SUM",
    data_format="NCHW",
):
    """
    Reconstructed image fromt extracted patches.
    See: https://stackoverflow.com/a/56891806

    :param patches: a 4-D Tensor in 'NCHW' or "NHWC" format where the depth
        dimension ('C') represents a patch
    :param img_size: a tuple (height, width) representing output image size
        (optional). If `None`, image size will be inferred from inputs.
    """
    with tf.name_scope("patches2im"):
        # To NHWC.
        if data_format == "NCHW":
            patches = patches.transpose([0, 2, 3, 1])

        _h = patches.shape[1]
        _w = patches.shape[2]

        bs = patches.shape[0]  # batch size
        np = _h * _w  # number of patches
        ps_h = patch_size[0]  # patch height
        ps_w = patch_size[1]  # patch width
        col_ch = patches.shape[3] // (ps_h * ps_w)  # Colour channel count
        assert patches.shape[3] % (ps_h * ps_w) == 0, "Unexpected patch size"

        patches = patches.reshape((bs, -1, ps_h, ps_w, col_ch))

        # Recalculate output shape of "extract_image_patches" including padded pixels
        wout = (_w - 1) * strides[0] + ps_w
        # Recalculate output shape of "extract_image_patches" including padded pixels
        hout = (_h - 1) * strides[1] + ps_h

        x, y = tf.meshgrid(tf.range(ps_w), tf.range(ps_h))
        x = tf.reshape(x, (1, 1, ps_h, ps_w, 1, 1))
        y = tf.reshape(y, (1, 1, ps_h, ps_w, 1, 1))
        xstart, ystart = tf.meshgrid(
            tf.range(0, (wout - ps_w) + 1, strides[0]),
            tf.range(0, (hout - ps_h) + 1, strides[1]),
        )

        bb = tf.zeros((1, np, ps_h, ps_w, col_ch, 1), dtype=tf.int32) + tf.reshape(
            tf.range(bs), (-1, 1, 1, 1, 1, 1)
        )  # batch indices
        yy = (
            tf.zeros((bs, 1, 1, 1, col_ch, 1), dtype=tf.int32)
            + y
            + tf.reshape(ystart, (1, -1, 1, 1, 1, 1))
        )  # y indices
        xx = (
            tf.zeros((bs, 1, 1, 1, col_ch, 1), dtype=tf.int32)
            + x
            + tf.reshape(xstart, (1, -1, 1, 1, 1, 1))
        )  # x indices
        cc = tf.zeros((bs, np, ps_h, ps_w, 1, 1), dtype=tf.int32) + tf.reshape(
            tf.range(col_ch), (1, 1, 1, 1, -1, 1)
        )  # color indices
        dd = tf.zeros((bs, 1, ps_h, ps_w, col_ch, 1), dtype=tf.int32) + tf.reshape(
            tf.range(np), (1, -1, 1, 1, 1, 1)
        )  # shift indices

        idx = tf.concat([bb, yy, xx, cc, dd], -1)

        stratified_img = patches.scatter_nd(idx, (bs, hout, wout, col_ch, np))
        stratified_img = stratified_img.transpose((0, 4, 1, 2, 3))

        stratified_img_count = patches.factory.ones_like(patches).scatter_nd(
            idx, (bs, hout, wout, col_ch, np)
        )
        stratified_img_count = stratified_img_count.transpose((0, 4, 1, 2, 3))

        with tf.name_scope("consolidate"):
            sum_stratified_img = stratified_img.reduce_sum(axis=1)
            if consolidation == "SUM":
                reconstructed_img = sum_stratified_img
            elif consolidation == "AVG":
                stratified_img_count = stratified_img_count.reduce_sum(axis=1)
                reconstructed_img = sum_stratified_img / stratified_img_count
            else:
                raise NotImplementedError(
                    "Unknown consolidation method: {}".format(consolidation)
                )

        if img_size is not None:
            img_h, img_w = img_size
            if img_h > hout:
                # This happens when padding is 'VALID' and the image has been cropped,
                # hence we will just pad 0 at the end
                reconstructed_img = reconstructed_img.factory.pad(
                    reconstructed_img, [[0, 0], [0, img_h - hout], [0, 0], [0, 0]]
                )
            elif img_h < hout:
                pad_top = (hout - img_h) // 2
                pad_bottom = hout - img_h - pad_top
                reconstructed_img = reconstructed_img[
                    :, pad_top : (hout - pad_bottom), :, :
                ]

            if img_w > wout:
                reconstructed_img = reconstructed_img.factory.pad(
                    reconstructed_img, [[0, 0], [0, 0], [0, img_w - hout], [0, 0]]
                )
            elif img_w < wout:
                pad_left = (wout - img_w) // 2
                pad_right = wout - img_w - pad_left
                reconstructed_img = reconstructed_img[
                    :, :, pad_left : (wout - pad_right), :
                ]

        # To NCHW.
        if data_format == "NCHW":
            reconstructed_img = reconstructed_img.transpose([0, 3, 1, 2])

        return reconstructed_img


def pad_size(input_size, kernel_size, strides):
    h_in, w_in = input_size
    if h_in % strides[0] == 0:
        pad_along_height = max(kernel_size[0] - strides[0], 0)
    else:
        pad_along_height = max(kernel_size[0] - (h_in % strides[0]), 0)

    if w_in % strides[1] == 0:
        pad_along_width = max(kernel_size[1] - strides[1], 0)
    else:
        pad_along_width = max(kernel_size[1] - (w_in % strides[1]), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return [[pad_top, pad_bottom], [pad_left, pad_right]]


def im2col(
    x,
    h_filter: int,
    w_filter: int,
    strides: list = [1, 1],
    padding: str = "SAME",
    data_format="NCHW",
) -> tf.Tensor:
    """Generic implementation of im2col on tf.Tensors."""

    with tf.name_scope("im2col"):

        if data_format == "NCHW":
            x = x.transpose([0, 2, 3, 1])
        channels = int(x.shape[3])

        patch_tensor = x.im2patches(
            (h_filter, w_filter),
            strides=strides,
            padding=padding,
            data_format="NHWC",
        )

        patch_tensor = patch_tensor.transpose([3, 1, 2, 0])
        patch_tensor = patch_tensor.reshape((h_filter, w_filter, channels, -1))

        if data_format == "NCHW":
            # Put channel first for each patch
            patch_tensor = patch_tensor.transpose([2, 0, 1, 3])

        # reshape to x_col
        x_col_tensor = patch_tensor.reshape(
            (channels * h_filter * w_filter, -1)
        )

        return x_col_tensor


def out_size(in_size, pool_size, strides, padding):

    if padding == "SAME":
        out_height = math.ceil(int(in_size[0]) / strides[0])
        out_width = math.ceil(int(in_size[1]) / strides[1])
    elif padding == "VALID":
        out_height = math.ceil((int(in_size[0]) - pool_size[0] + 1) / strides[0])
        out_width = math.ceil((int(in_size[1]) - pool_size[1] + 1) / strides[1])
    else:
        raise ValueError("Don't know padding method '{}'".format(padding))
    return [out_height, out_width]


def conv2d(x, y, strides=[1, 1], padding="SAME", data_format="NCHW"):
    """
    Generic convolution implementation with im2col over AbstractTensors.
    """

    with tf.name_scope("conv2d"):
        if data_format == "NCHW":
            x = x.transpose([0, 2, 3, 1])

        h_filter, w_filter, in_filters, out_filters = map(int, y.shape)
        n_x, h_x, w_x, c_x = map(int, x.shape)

        h_out, w_out = out_size([h_x, w_x], [h_filter, w_filter], strides, padding)

        x_col = im2col(
            x, h_filter, w_filter, strides=strides, padding=padding, data_format="NHWC"
        )
        w_col = y.transpose([3, 0, 1, 2]).reshape([int(out_filters), -1])
        out = w_col.matmul(x_col)

        out = out.reshape([out_filters, h_out, w_out, n_x])

        if data_format == "NCHW":
            out = out.transpose([3, 0, 1, 2])
        else:
            out = out.transpose([3, 1, 2, 0])

        return out
