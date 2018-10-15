from __future__ import absolute_import
from typing import Union, List, Tuple
from math import ceil

import numpy as np
import tensorflow as tf

from ..tensor.shared import im2col
from .helpers import inverse, prod

Decomposed = Union[List[tf.Tensor], List[np.ndarray]]


def gen_crt_decompose(m):

    def crt_decompose(x):

        with tf.name_scope('crt_decompose'):
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
                u = np.sum([ti * bi for ti, bi in zip(t, b)], axis=0).astype(np.int64)
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


def gen_crt_matmul(m):

    def crt_matmul(x, y):
        with tf.name_scope('crt_matmul'):
            return [tf.matmul(xi, yi) % mi for xi, yi, mi in zip(x, y, m)]

    return crt_matmul


def crt_matmul_split(
    x: Decomposed,
    y: Decomposed,
    threshold: int
) -> List[Tuple[Decomposed, Decomposed]]:

    with tf.name_scope('matmul_split'):
        z_split = []

        num_columns = int(x[0].shape[1])
        num_split = int(ceil(num_columns / threshold))
        for i in range(num_split):

            left = i * threshold
            right = (i + 1) * threshold

            inner_x = []  # type: List[Union[tf.Tensor, np.ndarray]]
            inner_y = []  # type: List[Union[tf.Tensor, np.ndarray]]

            for xi, yi in zip(x, y):
                inner_x.append(xi[:, left:right])
                inner_y.append(yi[left:right, :])

            z_split.append((inner_x, inner_y))

    return z_split


def crt_im2col(
    x: Decomposed,
    h_filter: int,
    w_filter: int,
    padding: str,
    strides: int
) -> Decomposed:
    with tf.name_scope('crt_im2col'):
        return [im2col(xi, h_filter, w_filter, padding, strides) for xi in x]


def gen_crt_sample_uniform(m, int_type):

    def crt_sample_uniform(shape):
        with tf.name_scope('sample'):
            return [tf.random_uniform(shape, maxval=mi, dtype=int_type) for mi in m]

    return crt_sample_uniform


def gen_crt_sample_bounded(m, int_type):

    CHUNK_MAX_BITLENGTH = 30  # TODO[Morten] bump to full range once signed numbers is settled (change minval etc)
    add = gen_crt_add(m)
    mul = gen_crt_mul(m)
    decompose = gen_crt_decompose(m)

    def crt_sample_bounded(shape, bitlength):

        with tf.name_scope('sample_bounded'):
            q, r = bitlength // CHUNK_MAX_BITLENGTH, bitlength % CHUNK_MAX_BITLENGTH
            chunk_sizes = [CHUNK_MAX_BITLENGTH] * q + ([r] if r > 0 else [])

            result = decompose(0)
            for chunk_size in chunk_sizes:
                chunk_value = tf.random_uniform(shape, minval=0, maxval=2**chunk_size, dtype=int_type)
                scale = 2**chunk_size
                result = add(
                    mul(result, decompose(scale)),
                    decompose(chunk_value)
                )

        return result

    return crt_sample_bounded


def gen_crt_mod(m, int_type):

    # outer precomputation
    M = prod(m)
    q = [inverse(M // mi, mi) for mi in m]

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
            return w % k

    return crt_mod


def gen_crt_reduce_sum(m):

    def crt_reduce_sum(x, axis=None, keepdims=None):

        with tf.name_scope('crt_reduce_sum'):
            return [tf.reduce_sum(xi, axis, keepdims) % mi for xi, mi in zip(x, m)]

    return crt_reduce_sum


def gen_crt_cumsum(m):

    def crt_cumsum(x, axis=None, exclusive=None, reverse=None):

        with tf.name_scope('crt_cumsum'):
            return [
                tf.cumsum(xi, axis=axis, exclusive=exclusive, reverse=reverse) % mi
                for xi, mi in zip(x, m)
            ]

    return crt_cumsum


def gen_crt_equal_zero(m, int_type):

    def crt_equal_zero(x, out_native_type):
        with tf.name_scope('crt_equal_zero'):
            number_of_zeros = tf.reduce_sum([tf.cast(tf.equal(xi, 0), out_native_type) for xi in x], axis=0)
            all_zeros = tf.cast(tf.equal(number_of_zeros, len(m)), out_native_type)
            return all_zeros

    return crt_equal_zero


def gen_crt_equal(m, int_type):

    def crt_equal(x, y, out_native_type):
        with tf.name_scope('crt_equal'):
            number_of_matches = tf.reduce_sum([tf.cast(tf.equal(xi, yi), out_native_type) for xi, yi in zip(x, y)], axis=0)
            all_matches = tf.cast(tf.equal(number_of_matches, len(m)), out_native_type)
            return all_matches

    return crt_equal


class CrtTensor(object):
    pass
