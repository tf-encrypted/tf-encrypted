from __future__ import absolute_import
from math import ceil

import numpy as np
import tensorflow as tf

from .helpers import inverse, prod
from typing import Union, List, Tuple

TFEData = Union[List[tf.Tensor], List[np.ndarray]]


def gen_crt_decompose(m):

    def crt_decompose(x):
        return tuple(x % mi for mi in m)

    return crt_decompose


def gen_crt_recombine_lagrange(m):

    # precomputation
    M = prod(m)
    n = [M // mi for mi in m]
    lambdas = [ni * inverse(ni, mi) % M for ni, mi in zip(n, m)]

    def crt_recombine_lagrange(x):

        with tf.name_scope('crt_recombine_lagrange'):
            res = sum(xi * li for xi, li in zip(x, lambdas)) % M
            res = res.astype(object)
            return res

    return crt_recombine_lagrange


def gen_crt_recombine_explicit(m, int_type):

    # precomputation
    M = prod(m)
    q = [inverse(M // mi, mi) for mi in m]

    def crt_recombine_explicit(x, bound):

        B = M % bound
        b = [(M // mi) % bound for mi in m]

        with tf.name_scope('crt_recombine_explicit'):

            if isinstance(x[0], np.ndarray):
                # backed by np.ndarray
                t = [(xi * qi) % mi for xi, qi, mi in zip(x, q, m)]
                alpha = np.round(
                    np.sum(
                        [ti.astype(float) / mi for ti, mi in zip(t, m)],
                        axis=0
                    ))
                u = np.sum((ti * bi for ti, bi in zip(t, b)), axis=0).astype(np.int64)
                v = alpha.astype(np.int64) * B
                w = u - v
                res = w % bound
                res = res.astype(np.int32)
                return res

            elif isinstance(x[0], tf.Tensor):
                # backed by tf.Tensor
                t = [(xi * qi) % mi for xi, qi, mi in zip(x, q, m)]
                alpha = tf.round(
                    tf.reduce_sum(
                        [tf.cast(ti, tf.float32) / mi for ti, mi in zip(t, m)],
                        axis=0
                    ))
                u = tf.cast(tf.reduce_sum([ti * bi for ti, bi in zip(t, b)], axis=0), tf.int64)
                v = tf.cast(alpha, tf.int64) * B
                w = u - v
                res = w % bound
                res = tf.cast(res, int_type)
                return res

            else:
                raise TypeError("Don't know how to recombine {}".format(type(x[0])))

    return crt_recombine_explicit


# *** NOTE ***
# keeping mod operations in-lined here for simplicity;
# we should do them lazily


def gen_crt_add(m):

    def crt_add(x, y):
        with tf.name_scope('crt_add'):
            return [(xi + yi) % mi for xi, yi, mi in zip(x, y, m)]

    return crt_add


def gen_crt_sub(m):

    def crt_sub(x, y):
        with tf.name_scope('crt_sub'):
            return [(xi - yi) % mi for xi, yi, mi in zip(x, y, m)]

    return crt_sub


def gen_crt_mul(m):

    def crt_mul(x, y):
        with tf.name_scope('crt_mul'):
            return [(xi * yi) % mi for xi, yi, mi in zip(x, y, m)]

    return crt_mul


def gen_crt_dot(m):

    def crt_dot(x, y):
        with tf.name_scope('crt_dot'):
            return [tf.matmul(xi, yi) % mi for xi, yi, mi in zip(x, y, m)]

    return crt_dot


def crt_matmul_split(x: TFEData, y: TFEData, threshold: int) -> List[Tuple[TFEData, TFEData]]:
    with tf.name_scope('matmul_split'):
        z_split = []

        num_columns = x[0].shape[1]
        num_split = int(ceil(num_columns / threshold))
        for i in range(num_split):

            left = i * threshold
            right = (i + 1) * threshold

            inner_x: List[Union[tf.Tensor, np.ndarray]] = []
            inner_y: List[Union[tf.Tensor, np.ndarray]] = []

            for xi, yi in zip(x, y):
                inner_x.append(xi[:, left:right])
                inner_y.append(yi[left:right, :])

            z_split.append((inner_x, inner_y))

    return z_split


def gen_crt_im2col(m):

    def im2col(x, h_filter, w_filter, padding, strides):
        with tf.name_scope('crt_im2col'):

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

            return x_col_tensors

    return im2col


def gen_crt_sample_uniform(m, int_type):

    def crt_sample_uniform(shape):
        with tf.name_scope('sample'):
            return [tf.random_uniform(shape, maxval=mi, dtype=int_type) for mi in m]

    return crt_sample_uniform


def gen_crt_mod(m, int_type):

    # outer precomputation
    M = prod(m)
    q = [inverse(M // mi, mi) for mi in m]
    redecompose = gen_crt_decompose(m)

    def crt_mod(x, k):
        assert type(k) in [int], type(k)

        # inner precomputations
        B = M % k
        b = [(M // mi) % k for mi in m]

        with tf.name_scope('crt_mod'):
            t = [(xi * qi) % mi for xi, qi, mi in zip(x, q, m)]
            alpha = tf.round(
                tf.reduce_sum(
                    [tf.cast(ti, tf.float32) / mi for ti, mi in zip(t, m)],
                    axis=0
                )
            )
            u = tf.reduce_sum([ti * bi for ti, bi in zip(t, b)], axis=0)
            v = tf.cast(alpha, int_type) * B
            w = u - v
            return redecompose(w % k)

    return crt_mod


def gen_crt_sum(m):
    def crt_sum(x, axis, keepdims=None):
        with tf.name_scope('crt_sum'):
            dims = x[0].shape.dims.copy()
            ax_len = dims.pop(axis)
            begins = [0] * len(dims)
            ends = [x.value for x in dims]
            perm = [axis, *[i for i in range(len(dims) + 1) if i != axis]]
            x = [tf.transpose(xi, perm=perm) for xi in x]
            y = [tf.zeros(dims, dtype=xi.dtype) for xi in x]
            for i in range(ax_len):
                for j in range(len(x)):
                    sl = x[j][i]
                    y[j] += sl
                    y[j] %= m[j]
                begins[0] += 1
                ends[0] += 1
            if keepdims:
                return [tf.expand_dims(yi, axis) for yi in y]
            return y

    return crt_sum


class CrtTensor(object):
    pass
