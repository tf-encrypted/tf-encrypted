import numpy as np
import tensorflow as tf

from helpers import inverse, prod, log2

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

        # inner precomputations
        K = 2**k
        B = M % K
        b = [ (M // mi) % K for mi in m ]

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
            return redecompose(v % K)

    return crt_mod
