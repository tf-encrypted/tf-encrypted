from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from .helpers import inverse, prod, log2

def gen_crt_decompose(m):

    def crt_decompose(x):
        return tuple( x % mi for mi in m )

    return crt_decompose

def gen_crt_recombine(m, lambdas):
    # TODO compute lambdas based on m

    # precomputation
    M = prod(m)

    def crt_recombine(x):
        return sum( xi * li for xi, li in zip(x, lambdas) ) % M

    return crt_recombine

# *** NOTE ***
# keeping mod operations in-lined here for simplicity;
# we should do them lazily

def gen_crt_add(m):

    def crt_add(x, y):
        with tf.name_scope('crt_add'):
            return [ (xi + yi) % mi for xi, yi, mi in zip(x, y, m) ]

    return crt_add

def gen_crt_sub(m):

    def crt_sub(x, y):
        with tf.name_scope('crt_sub'):
            return [ (xi - yi) % mi for xi, yi, mi in zip(x, y, m) ]

    return crt_sub

def gen_crt_mul(m):

    def crt_mul(x, y):
        with tf.name_scope('crt_mul'):
            return [ (xi * yi) % mi for xi, yi, mi in zip(x, y, m) ]

    return crt_mul

def gen_crt_dot(m):

    def crt_dot(x, y):
        with tf.name_scope('crt_dot'):
            return [ tf.matmul(xi, yi) % mi for xi, yi, mi in zip(x, y, m) ]

    return crt_dot

def gen_crt_im2col(m):

    def im2col(x, h_filter, w_filter, padding, strides):
        with tf.name_scope('crt_im2col'):

            # we need NHWC because tf.extract_image_patches expects this
            NHWC_tensors = [tf.transpose(xi, [0, 2, 3, 1]) for xi in x]
            channels = int(NHWC_tensors[0].shape[3])
            # extract patches
            patch_tensors = [tf.extract_image_patches(xi,
                                              ksizes=[1, h_filter, w_filter, 1],
                                              strides=[1, strides, strides, 1],
                                              rates=[1, 1, 1, 1],
                                              padding=padding) for xi in NHWC_tensors]
            # change back to NCHW
            patch_tensors_NCHW = [tf.reshape(tf.transpose(patches, [3, 1, 2, 0]), (h_filter, w_filter, channels, -1))
                                  for patches in patch_tensors]
            # reshape to x_col
            x_col_tensors = [tf.reshape(tf.transpose(x_col_NHWC, [2, 0, 1, 3]), (channels * h_filter * w_filter, -1))
                             for x_col_NHWC in patch_tensors_NCHW]
            return x_col_tensors

    return im2col

def gen_crt_sample_uniform(m, int_type):

    def crt_sample_uniform(shape):
        with tf.name_scope('sample'):
            return [ tf.random_uniform(shape, maxval=mi, dtype=int_type) for mi in m ]

    return crt_sample_uniform

def gen_crt_mod(m, int_type, float_type):

    # outer precomputation
    M = prod(m)
    q = [ inverse(M // mi, mi) for mi in m ]
    redecompose = gen_crt_decompose(m)

    def crt_mod(x, k):
        assert type(k) in [int], type(k)

        # inner precomputations
        B = M % k
        b = [ (M // mi) % k for mi in m ]

        with tf.name_scope('crt_mod'):
            t = [ (xi * qi) % mi for xi, qi, mi in zip(x, q, m) ]
            alpha = tf.cast(
                tf.round(
                    tf.reduce_sum(
                        [ tf.cast(ti, float_type) / mi for ti, mi in zip(t, m) ],
                        axis=0
                    )
                ),
                int_type
            )
            v = tf.reduce_sum(
                [ ti * bi for ti, bi in zip(t, b) ],
                axis=0
            ) - B * alpha
            return redecompose(v % k)

    return crt_mod


def gen_crt_sum(m):
    def crt_sum(x, axis, keepdims=None):
        with tf.name_scope('crt_sum'):
            dims = x.shape.dims
            begins = [0] * len(dims)
            ends = [1] + [0] * (len(dims) - 1)
            ax_len = dims.pop(axis)
            x = tf.transpose(x, perm=[axis, *dims])
            y = tf.zeros(dims, dtype=x.dtype)
            for i in range(ax_len):
                y += tf.slice(x, begins, ends)
                y %= m
                begins[0] += 1
                ends[0] += 1
            if keepdims:
                return tf.expand_dims(y, axis)
            return y

        return crt_sum

class CrtTensor(object):
    pass
