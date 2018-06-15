
# TODOs:
# - use TensorArray as training buffers instead of FIFOQueue
# - recombine without blowing up numbers (should fit in 64bit word)
# - gradient computation + SGD
# - compare performance if native type is float64 instead of int64
# - performance on GPU
# - better cache strategy?
# - does it make sense to cache additions, subtractions, etc as well?
# - make truncation optional; should work even with cached results
# - lazy mods
# - sigmoid() is showing some unused substructures in TensorBoard; why?

from functools import reduce
from math import log

import numpy as np
import tensorflow as tf

from crt import *

log2 = lambda x: log(x)/log(2)
prod = lambda xs: reduce(lambda x,y: x*y, xs)

from protocol import get_active_protocol

from config import (
    session, run
)

# Idea is to simulate five different players on different devices. 
# Hopefully Tensorflow can take care of (optimising?) networking like this.

#
# 64 bit CRT
# - 5 components for modulus ~120 bits (encoding 16.32)
#
# BITPRECISION_INTEGRAL   = 16
# BITPRECISION_FRACTIONAL = 30
# INT_TYPE = tf.int64
# FLOAT_TYPE = tf.float64
# TRUNCATION_GAP = 20
# m = [89702869, 78489023, 69973811, 70736797, 79637461]
# M = 2775323292128270996149412858586749843569 # == prod(m)
# lambdas = [
#     875825745388370376486957843033325692983, 
#     2472444909335399274510299476273538963924, 
#     394981838173825602426191615931186905822, 
#     2769522470404025199908727961712750149119, 
#     1813194913083192535116061678809447818860
# ]

#
# 32 bit CRT
# - we need this to do dot product as int32 is the only supported type for that
# - tried tf.float64 but didn't work out of the box
# - 10 components for modulus ~100 bits
#
BITPRECISION_INTEGRAL   = 16
BITPRECISION_FRACTIONAL = 16
INT_TYPE = tf.int32
FLOAT_TYPE = tf.float32
TRUNCATION_GAP = 20
m = [1201, 1433, 1217, 1237, 1321, 1103, 1129, 1367, 1093, 1039]
M = 6616464272061971915798970247351 # == prod(m)
lambdas = [
    1008170659273389559193348505633, 
    678730110253391396805616626909, 
    3876367317978788805229799331439, 
    1733010852181147049893990590252, 
    2834912019672275627813941831946, 
    5920625781074493455025914446179, 
    4594604064921688203708053741296, 
    4709451160728821268524065874669,
    4618812662015813880836792588041, 
    3107636732210050331963327700392
]

for mi in m: assert 2*log2(mi) + log2(1024) < log2(INT_TYPE.max)
assert log2(M) >= 2 * (BITPRECISION_INTEGRAL + BITPRECISION_FRACTIONAL) + log2(1024) + TRUNCATION_GAP

K = 2 ** BITPRECISION_FRACTIONAL

def _egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = _egcd(b % a, a)
        return (g, x - (b // a) * y, y)
    
def _gcd(a, b):
    g, _, _ = _egcd(a, b)
    return g

def _inverse(a, m):
    _, b, _ = _egcd(a, m)
    return b % m

def encode(rationals, precision=BITPRECISION_FRACTIONAL):
    """ [NumPy] Encode tensor of rational numbers into tensor of ring elements """
    return (rationals * (2**precision)).astype(int).astype(object) % M

def decode(elements, precision=BITPRECISION_FRACTIONAL):
    """ [NumPy] Decode tensor of ring elements into tensor of rational numbers """
    map_negative_range = np.vectorize(lambda element: element if element <= M/2 else element - M)
    return map_negative_range(elements).astype(float) / (2**precision)

def decompose(x):
    return tuple( x % mi for mi in m )

def recombine(x):
    return sum( xi * li for xi, li in zip(x, lambdas) ) % M

# *** NOTE ***
# keeping mod operations in-lined here for simplicity;
# we should do them lazily

def crt_add(x, y):
    with tf.name_scope('crt_add'):
        return [ (xi + yi) % mi for xi, yi, mi in zip(x, y, m) ]

def crt_sub(x, y):
    with tf.name_scope("crt_sub"):
        return [ (xi - yi) % mi for xi, yi, mi in zip(x, y, m) ]

def crt_scale(x, k):
    with tf.name_scope("crt_scale"):
        return [ (xi * ki) % mi for xi, ki, mi in zip(x, k, m) ]

def crt_mul(x, y):
    with tf.name_scope("crt_mul"):
        return [ (xi * yi) % mi for xi, yi, mi in zip(x, y, m) ]

def crt_dot(x, y):
    with tf.name_scope("crt_dot"):
        return [ tf.matmul(xi, yi) % mi for xi, yi, mi in zip(x, y, m) ]

def _gen_crt_mod():

    # precomputation
    q = [ _inverse(M // mi, mi) for mi in m ]
    B = M % K
    b = [ (M // mi) % K for mi in m ]

    def crt_mod(x):
        with tf.name_scope("crt_mod"):
            t = [ (xi * qi) % mi for xi, qi, mi in zip(x, q, m) ]
            alpha = tf.cast(
                tf.round(
                    tf.reduce_sum(
                        [ tf.cast(ti, FLOAT_TYPE) / mi for ti, mi in zip(t, m) ],
                        axis=0
                    )
                ),
                INT_TYPE
            )
            v = tf.reduce_sum(
                [ ti * bi for ti, bi in zip(t, b) ],
                axis=0
            ) - B * alpha
            return decompose(v % K)

    return crt_mod

crt_mod = _gen_crt_mod()

def sample(shape):
    with tf.name_scope('sample'):
        return [ tf.random_uniform(shape, maxval=mi, dtype=INT_TYPE) for mi in m ]

def share(secret):
    with tf.name_scope('share'):
        shape = secret[0].shape
        share0 = sample(shape)
        share1 = crt_sub(secret, share0)
        return share0, share1

def reconstruct(share0, share1):
    with tf.name_scope('reconstruct'):
        return crt_add(share0, share1)

_nodes = dict()

class Tensor(object):

    def __init__(self, xs):
        self.xs = xs

    @property
    def shape(self):
        return self.xs[0].shape

    @property
    def unwrapped(self):
        return self.xs

class PrivateTensor(object):
    
    def __init__(self, share0, share1):
        self.share0 = share0
        self.share1 = share1
    
    @property
    def shape(self):
        return self.share0[0].shape

    @property
    def unwrapped(self):
        return (self.share0, self.share1)

    def __add__(x, y):
        return add(x, y)
    
    def __sub__(x, y):
        return sub(x, y)
    
    def __mul__(x, y):
        return mul(x, y)

    def dot(x, y):
        return dot(x, y)

    def truncate(x):
        return truncate(x)

class MaskedPrivateTensor(object):

    def __init__(self, x, a, a0, a1, alpha_on_0, alpha_on_1):
        assert isinstance(x, PrivateTensor)
        self.x  = x
        self.a  = a
        self.a0 = a0
        self.a1 = a1
        self.alpha_on_0 = alpha_on_0
        self.alpha_on_1 = alpha_on_1

    @property
    def shape(self):
        return self.a[0].shape

    @property
    def unmasked(self):
        return self.x

    @property
    def unwrapped(self):
        return (self.a, self.a0, self.a1, self.alpha_on_0, self.alpha_on_1)

def transpose(x):
    assert isinstance(x, PrivateTensor)

    x0, x1 = x.share0, x.share1

    with tf.name_scope('transpose'):

        with tf.device(get_active_protocol().server_0.device_name):
            x0_t = [ tf.transpose(t) for t in x0 ]

        with tf.device(get_active_protocol().server_1.device_name):
            x1_t = [ tf.transpose(t) for t in x1 ]

        x_t = PrivateTensor(x0_t, x1_t)

        x_masked = _nodes.get(('mask', x), None)
        if x_masked:
            # use mask for `x` to get mask for `x_t`

            a, a0, a1, alpha_on_0, alpha_on_1 = x_masked.unwrapped

            with tf.device(get_active_protocol().crypto_producer.device_name):
                a_t = [ tf.transpose(t) for t in a ]

            with tf.device(get_active_protocol().server_0.device_name):
                a0_t = [ tf.transpose(t) for t in a0 ]
                alpha_on_0_t = [ tf.transpose(t) for t in alpha_on_0 ]

            with tf.device(get_active_protocol().server_1.device_name):
                a1_t = [ tf.transpose(t) for t in a1 ]
                alpha_on_1_t = [ tf.transpose(t) for t in alpha_on_1 ]

            x_masked_t = MaskedPrivateTensor(x_t, a_t, a0_t, a1_t, alpha_on_0_t, alpha_on_1_t)
            _nodes[('mask', x_t)] = x_masked_t

    return x_t

def concat(ys):
    # FIXME[Morten] add support for PrivateTensors as well

    def helper(tensors):
        # as an example, assume shape is [[(1000,2); 10]; 3]
        tensors = tf.concat(tensors, axis=1)
        # now shape is (10,3000,2)
        tensors = tf.split(tensors, 10, axis=0)
        # now shape is [(1,3000,2); 10]
        tensors = [ tf.reshape(tensor, tensor.shape[1:]) for tensor in tensors ]
        # now shape is [(3000,2); 10]
        return tensors

    with tf.name_scope('concat'):

        y0s, y1s = zip(*[ y.unmasked.unwrapped for y in ys ])
        bs, b0s, b1s, beta_on_0s, beta_on_1s = zip(*[ y.unwrapped for y in ys ])

        with tf.device(get_active_protocol().crypto_producer.device_name):
            b = helper(bs)

        with tf.device(get_active_protocol().server_0.device_name):
            y0 = helper(y0s)
            b0 = helper(b0s)
            beta_on_0 = helper(beta_on_0s)

        with tf.device(get_active_protocol().server_1.device_name):
            y1 = helper(y1s)
            b1 = helper(b1s)
            beta_on_1 = helper(beta_on_1s)

        y = PrivateTensor(y0, y1)
        y_masked = MaskedPrivateTensor(y, b, b0, b1, beta_on_0, beta_on_1)

    return y_masked

def split(y, num_splits):
    assert isinstance(y, MaskedPrivateTensor)
    # FIXME[Morten] add support for PrivateTensors as well

    y0, y1 = y.unmasked.unwrapped
    b, b0, b1, beta_on_0, beta_on_1 = y.unwrapped

    def helper(tensors):
        # FIXME[Morten] all this reshaping seems to encur a big hit on (at least) graph building
        # as an example, assume shape is [(3000,2); 10]
        tensors = tf.stack(tensors)
        # now shape is (10,3000,2)
        tensors = tf.split(tensors, num_splits, axis=1)
        # now shape is [(10,30,2); 100] if num_splits == 100
        tensors = [ 
            [ tf.reshape(xi, xi.shape[1:]) for xi in tf.split(tensor, 10, axis=0) ]
            for tensor in tensors
        ]
        # now shape is [[(30,2); 10]; 100]
        return tensors

    with tf.name_scope('split'):

        with tf.device(get_active_protocol().crypto_producer.device_name):
            bs = helper(b)

        with tf.device(get_active_protocol().server_0.device_name):
            y0s = helper(y0)
            b0s = helper(b0)
            beta_on_0s = helper(beta_on_0)

        with tf.device(get_active_protocol().server_1.device_name):
            y1s = helper(y1)
            b1s = helper(b1)
            beta_on_1s = helper(beta_on_1)

    tensors = []
    for y0, y1, b, b0, b1, beta_on_0, beta_on_1 in zip(y0s, y1s, bs, b0s, b1s, beta_on_0s, beta_on_1s):
        y = PrivateTensor(y0,y1)
        y_masked = MaskedPrivateTensor(y, b, b0, b1, beta_on_0, beta_on_1)
        tensors.append(y_masked)
    
    return tensors

def add(x, y):
    assert isinstance(x, PrivateTensor)
    assert isinstance(y, PrivateTensor)
    
    node_key = ('add', x, y)
    z = _nodes.get(node_key, None)

    if z is None:

        x0, x1 = x.unwrapped
        y0, y1 = y.unwrapped
        
        with tf.name_scope('add'):
        
            with tf.device(get_active_protocol().server_0.device_name):
                z0 = crt_add(x0, y0)

            with tf.device(get_active_protocol().server_1.device_name):
                z1 = crt_add(x1, y1)

        z = PrivateTensor(z0, z1)
        _nodes[node_key] = z

    return z

def sub(x, y):
    assert isinstance(x, PrivateTensor)
    assert isinstance(y, PrivateTensor)
    
    node_key = ('sub', x, y)
    z = _nodes.get(node_key, None)

    if z is None:

        x0, x1 = x.unwrapped
        y0, y1 = y.unwrapped
        
        with tf.name_scope("sub"):
        
            with tf.device(get_active_protocol().server_0.device_name):
                z0 = crt_sub(x0, y0)

            with tf.device(get_active_protocol().server_1.device_name):
                z1 = crt_sub(x1, y1)

        z = PrivateTensor(z0, z1)
        _nodes[node_key] = z

    return z

def scale(x, k, apply_encoding=None):
    assert isinstance(x, PrivateTensor)
    assert type(k) in [int, float]

    x0, x1 = x.unwrapped

    if apply_encoding is None:
        # determine automatically
        apply_encoding = type(k) is float

    c = np.array([k])
    if apply_encoding: c = encode(c)
    c = decompose(c)

    with tf.name_scope('scale'):

        with tf.device(get_active_protocol().server_0.device_name):
            y0 = crt_scale(x0, c)

        with tf.device(get_active_protocol().server_1.device_name):
            y1 = crt_scale(x1, c)

    y = PrivateTensor(y0, y1)
    if apply_encoding:
        y = truncate(y)

    return y

def mask(x):
    assert isinstance(x, PrivateTensor)
    
    node_key = ('mask', x)
    masked = _nodes.get(node_key, None)

    if masked is None:
      
        x0, x1 = x.unwrapped
        shape = x.shape
      
        with tf.name_scope('mask'):

            with tf.device(get_active_protocol().crypto_producer.device_name):
                a = sample(shape)
                a0, a1 = share(a)

            with tf.device(get_active_protocol().server_0.device_name):
                alpha0 = crt_sub(x0, a0)

            with tf.device(get_active_protocol().server_1.device_name):
                alpha1 = crt_sub(x1, a1)

            # exchange of alphas

            with tf.device(get_active_protocol().server_0.device_name):
                alpha_on_0 = reconstruct(alpha0, alpha1)

            with tf.device(get_active_protocol().server_1.device_name):
                alpha_on_1 = reconstruct(alpha0, alpha1)

        masked = MaskedPrivateTensor(x, a, a0, a1, alpha_on_0, alpha_on_1)
        _nodes[node_key] = masked
        
    return masked

def local_mask(x):
    assert isinstance(x, Tensor), type(x)

    with tf.name_scope('local_mask'):
        x0, x1 = share(x.unwrapped)
        a = sample(x.shape)
        a0, a1 = share(a)
        alpha = crt_sub(x.unwrapped, a)

    return MaskedPrivateTensor(PrivateTensor(x0, x1), a, a0, a1, alpha, alpha)

global_cache_updators = []

def cache(x, initializers=None, updators=None):

    if updators is None:
        updators = global_cache_updators

    # TODO[Morten] use `initializers`

    node_key = ('cache', x)
    cached = _nodes.get(node_key, None)

    if cached is None:

        if isinstance(x, PrivateTensor):

            x0, x1 = x.unwrapped

            with tf.name_scope('cache'):

                with tf.device(get_active_protocol().server_0.device_name):
                    cached_x0 = [ tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(x0, m) ]
                    updators.append([ tf.assign(var, val) for var, val in zip(cached_x0, x0) ])

                with tf.device(get_active_protocol().server_1.device_name):
                    cached_x1 = [ tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(x1, m) ]
                    updators.append([ tf.assign(var, val) for var, val in zip(cached_x1, x1) ])

            # TODO[Morten] wrap PrivateTensor around var.read_value() instead to ensure updated values?
            cached = PrivateTensor(cached_x0, cached_x1)
            _nodes[node_key] = cached

        elif isinstance(x, MaskedPrivateTensor):

            a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped
            cached_x = cache(x.unmasked, initializers, updators)

            with tf.name_scope('cache'):

                with tf.device(get_active_protocol().crypto_producer.device_name):
                    cached_a = [ tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(a, m) ]
                    updators.append([ tf.assign(var, val) for var, val in zip(cached_a, a) ])

                with tf.device(get_active_protocol().server_0.device_name):
                    cached_a0 = [ tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(a0, m) ]
                    updators.append([ tf.assign(var, val) for var, val in zip(cached_a0, a0) ])

                    cached_alpha_on_0 = [ tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(alpha_on_0, m) ]
                    updators.append([ tf.assign(var, val) for var, val in zip(cached_alpha_on_0, alpha_on_0) ])

                with tf.device(get_active_protocol().server_1.device_name):
                    cached_a1 = [ tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(a1, m) ]
                    updators.append([ tf.assign(var, val) for var, val in zip(cached_a1, a1) ])

                    cached_alpha_on_1 = [ tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(alpha_on_1, m) ]
                    updators.append([ tf.assign(var, val) for var, val in zip(cached_alpha_on_1, alpha_on_1) ])

            # TODO[Morten] wrap MaskedPrivateTensor around var.read_value() instead to ensure updated values?
            cached = MaskedPrivateTensor(
                cached_x,
                cached_a,
                cached_a0,
                cached_a1,
                cached_alpha_on_0,
                cached_alpha_on_1
            )
            _nodes[node_key] = cached

        else:
            raise AssertionError("'x' not of supported type")

    return cached

def square(x):

    node_key = ('square', x)
    y = _nodes.get(node_key, None)

    if y is None:

        if isinstance(x, PrivateTensor):
            x = mask(x)

        assert isinstance(x, MaskedPrivateTensor)
        a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped

        with tf.name_scope('square'):

            with tf.device(get_active_protocol().crypto_producer.device_name):
                aa = crt_mul(a, a)
                aa0, aa1 = share(aa)

            with tf.device(get_active_protocol().server_0.device_name):
                alpha = alpha_on_0
                y0 = crt_add(aa0,
                     crt_add(crt_mul(a0, alpha),
                     crt_add(crt_mul(alpha, a0), # TODO replace with `scale(, 2)` op
                             crt_mul(alpha, alpha))))

            with tf.device(get_active_protocol().server_1.device_name):
                alpha = alpha_on_1
                y1 = crt_add(aa1,
                     crt_add(crt_mul(a1, alpha),
                             crt_mul(alpha, a1))) # TODO replace with `scale(, 2)` op
        
        y = PrivateTensor(y0, y1)
        y = truncate(y)
        _nodes[node_key] = y

    return y

def mul(x, y):

    node_key = ('mul', x, y)
    z = _nodes.get(node_key, None)

    if z is None:

        if isinstance(x, PrivateTensor):
            x = mask(x)

        if isinstance(y, PrivateTensor):
            y = mask(y)

        assert isinstance(x, MaskedPrivateTensor)
        assert isinstance(y, MaskedPrivateTensor)

        a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped
        b, b0, b1,  beta_on_0,  beta_on_1 = y.unwrapped

        with tf.name_scope('mul'):

            with tf.device(get_active_protocol().crypto_producer.device_name):
                ab = crt_mul(a, b)
                ab0, ab1 = share(ab)

            with tf.device(get_active_protocol().server_0.device_name):
                alpha = alpha_on_0
                beta = beta_on_0
                z0 = crt_add(ab0,
                     crt_add(crt_mul(a0, beta),
                     crt_add(crt_mul(alpha, b0),
                             crt_mul(alpha, beta))))

            with tf.device(get_active_protocol().server_1.device_name):
                alpha = alpha_on_1
                beta = beta_on_1
                z1 = crt_add(ab1,
                     crt_add(crt_mul(a1, beta),
                             crt_mul(alpha, b1)))
        
        z = PrivateTensor(z0, z1)
        z = truncate(z)
        _nodes[node_key] = z

    return z

def dot(x, y):

    node_key = ('dot', x, y)
    z = _nodes.get(node_key, None)

    if z is None:

        if isinstance(x, PrivateTensor):
            x = mask(x)

        if isinstance(y, PrivateTensor):
            y = mask(y)

        assert isinstance(x, MaskedPrivateTensor)
        assert isinstance(y, MaskedPrivateTensor)
        
        a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped
        b, b0, b1,  beta_on_0,  beta_on_1 = y.unwrapped

        with tf.name_scope('dot'):

            with tf.device(get_active_protocol().crypto_producer.device_name):
                ab = crt_dot(a, b)
                ab0, ab1 = share(ab)

            with tf.device(get_active_protocol().server_0.device_name):
                alpha = alpha_on_0
                beta = beta_on_0
                z0 = crt_add(ab0,
                     crt_add(crt_dot(a0, beta),
                     crt_add(crt_dot(alpha, b0),
                             crt_dot(alpha, beta))))

            with tf.device(get_active_protocol().server_1.device_name):
                alpha = alpha_on_1
                beta = beta_on_1
                z1 = crt_add(ab1,
                     crt_add(crt_dot(a1, beta),
                             crt_dot(alpha, b1)))

        z = PrivateTensor(z0, z1)
        z = truncate(z)
        _nodes[node_key] = z

    return z

def _gen_truncate():
    assert _gcd(K, M) == 1

    # precomputation for truncation
    K_inv = decompose(_inverse(K, M))
    M_wrapped = decompose(M)

    def raw_truncate(x):
        y = crt_sub(x, crt_mod(x))
        return crt_mul(y, K_inv)

    def truncate(x):
        assert isinstance(x, PrivateTensor)

        x0, x1 = x.share0, x.share1

        with tf.name_scope('truncate'):
    
            with tf.device(get_active_protocol().server_0.device_name):
                y0 = raw_truncate(x0)

            with tf.device(get_active_protocol().server_1.device_name):
                y1 = crt_sub(M_wrapped, raw_truncate(crt_sub(M_wrapped, x1)))

        return PrivateTensor(y0, y1)

    return truncate

truncate = _gen_truncate()

def sigmoid(x):
    assert isinstance(x, PrivateTensor)

    w0 =  0.5
    w1 =  0.2159198015
    w3 = -0.0082176259
    w5 =  0.0001825597
    w7 = -0.0000018848
    w9 =  0.0000000072

    with tf.name_scope('sigmoid'):

        # TODO optimise depth
        x2 = square(x)
        x3 = mul(x2, x)
        x5 = mul(x2, x3)
        x7 = mul(x2, x5)
        x9 = mul(x2, x7)

        y1 = scale(x,  w1)
        y3 = scale(x3, w3)
        y5 = scale(x5, w5)
        y7 = scale(x7, w7)
        y9 = scale(x9, w9)

        with tf.device(get_active_protocol().server_0.device_name):
            z0 = crt_add(y1.share0,
                 crt_add(y3.share0,
                 crt_add(y5.share0,
                 crt_add(y7.share0,
                 crt_add(y9.share0,
                         decompose(encode(np.array([w0]))))))))

        with tf.device(get_active_protocol().server_1.device_name):
            z1 = crt_add(y1.share1,
                 crt_add(y3.share1,
                 crt_add(y5.share1,
                 crt_add(y7.share1,
                         y9.share1))))

    z = PrivateTensor(z0, z1)
    return z

def define_input(shape, name=None):
    
    with tf.name_scope('input{}'.format('-'+name if name else '')):
        
        with tf.device(INPUT_PROVIDER):
            input_x = [ tf.placeholder(INT_TYPE, shape=shape) for _ in m ]
            x0, x1 = share(input_x)
        
    return input_x, PrivateTensor(x0, x1)

def define_variable(initial_value, apply_encoding=True, name=None):
    
    v = initial_value
    v = encode(v) if apply_encoding else v
    v = decompose(v)
    v0, v1 = share(v)

    with tf.name_scope('var{}'.format('-'+name if name else '')):

        with tf.device(get_active_protocol().server_0.device_name):
            vars0 = [ tf.Variable(vi, dtype=INT_TYPE) for vi in v0 ]
            init0 = [ vi.initializer for vi in vars0 ]
            x0    = [ vi.read_value() for vi in vars0 ]

        with tf.device(get_active_protocol().server_1.device_name):
            vars1 = [ tf.Variable(vi, dtype=INT_TYPE) for vi in v1 ]
            init1 = [ vi.initializer for vi in vars1 ]
            x1    = [ vi.read_value() for vi in vars1 ]

        x = PrivateTensor(x0, x1)

    return x, init0+init1

def assign(x, v):
    assert isinstance(x, PrivateTensor)
    assert isinstance(v, PrivateTensor)

    x0, x1 = x.share0, x.share1
    v0, v1 = v.share0, v.share1

    with tf.name_scope("assign"):

        with tf.device(get_active_protocol().server_0.device_name):
            y0 = [ tf.assign(xi, vi) for xi, vi in zip(x0, v0) ]

        with tf.device(get_active_protocol().server_1.device_name):
            y1 = [ tf.assign(xi, vi) for xi, vi in zip(x1, v1) ]

    return y0, y1

def reveal(x):
    assert isinstance(x, PrivateTensor)
    
    x0, x1 = x.share0, x.share1

    with tf.name_scope('reveal'):
    
        with tf.device(OUTPUT_RECEIVER):
            y = reconstruct(x0, x1)
    
    return y

def encode_input(vars_and_values):
    if not isinstance(vars_and_values, list):
        vars_and_values = [vars_and_values]
    result = dict()
    for input_x, X in vars_and_values:
        result.update( (input_xi, Xi) for input_xi, Xi in zip(input_x, decompose(encode(X))) )
    return result

def decode_output(value):
    return decode(recombine(value))

def pyfunc_hack(func, x, shape=None):
    """ Essentially calls `tf.py_func(func, [x])` but supports
    returning (lists of) ints as needed by e.g. encoding. """

    # TODO[Morten]
    # tf.py_func currently has some limitations that mean we can't
    # directly apply it to e.g. encoding and decomposition;
    # concretely, it doesn't allow passing of ints.
    # this hack gets around that limitation by converting to
    # floats when exiting pyfunc, as well as some packing to
    # account for working with lists of tensors.

    # patch function to apply packing before returning
    def patched_func(x):
        y = func(x)
        y = pyfunc_hack_preexit(y)
        return y

    # applied just before exiting tf.py_func
    def pyfunc_hack_preexit(x):
        assert type(x) in [tuple, list]
        for xi in x: assert type(xi) == np.ndarray
        # convert all values to floats; at least for small 
        # ints this should give correct results
        x = [ xi.astype(float) for xi in x ]
        # pack list into single tensor
        x = np.array(x)
        return x

    # applied just after exiting tf.py_func
    def pyfunc_hack_postexit(x, component_shape, num_components=10):
        assert type(x) == tf.Tensor
        # unpack tensor into list
        x = [
            tf.reshape(xi, component_shape)
            for xi in tf.split(x, num_components)
        ]
        # convert to ints
        x = [ tf.cast(xi, dtype=tf.int32) for xi in x ]
        return x

    y = tf.py_func(patched_func, [x], tf.double)
    y = pyfunc_hack_postexit(y, shape or x.shape)
    return y

def encode_and_decompose(x, shape=None):
    func = lambda x: decompose(encode(x))
    return pyfunc_hack(func, x, shape=shape)

def recombine_and_decode(x):
    return decode(recombine(x))
