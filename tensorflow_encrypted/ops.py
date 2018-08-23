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

from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from .tensor.int100 import Int100Tensor
from .protocol.protocol import get_protocol


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

# *** NOTE ***
# keeping mod operations in-lined here for simplicity;
# we should do them lazily

# TODO[Morten] how to support this one with new abstractions?
def concat(ys):
    # FIXME[Morten] add support for PrivateTensors as well

    def helper(tensors):
        # as an example, assume shape is [[(1000,2); 10]; 3]
        tensors = tf.concat(tensors, axis=1)
        # now shape is (10,3000,2)
        tensors = tf.split(tensors, 10, axis=0)
        # now shape is [(1,3000,2); 10]
        tensors = [tf.reshape(tensor, tensor.shape[1:]) for tensor in tensors]
        # now shape is [(3000,2); 10]
        return tensors

    with tf.name_scope('concat'):

        y0s, y1s = zip(*[yunmasked.unwrapped for y in ys])
        bs, b0s, b1s, beta_on_0s, beta_on_1s = zip(*[y.unwrapped for y in ys])

        with tf.device(get_protocol().crypto_producer.device_name):
            b = helper(bs)

        with tf.device(get_protocol().server_0.device_name):
            y0 = helper(y0s)
            b0 = helper(b0s)
            beta_on_0 = helper(beta_on_0s)

        with tf.device(get_protocol().server_1.device_name):
            y1 = helper(y1s)
            b1 = helper(b1s)
            beta_on_1 = helper(beta_on_1s)

        y = PrivateTensor(y0, y1)
        y_masked = MaskedPrivateTensor(y, b, b0, b1, beta_on_0, beta_on_1)

    return y_masked

# TODO[Morten] how to support this one with new abstractions?


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
            [tf.reshape(xi, xi.shape[1:]) for xi in tf.split(tensor, 10, axis=0)]
            for tensor in tensors
        ]
        # now shape is [[(30,2); 10]; 100]
        return tensors

    with tf.name_scope('split'):

        with tf.device(get_protocol().crypto_producer.device_name):
            bs = helper(b)

        with tf.device(get_protocol().server_0.device_name):
            y0s = helper(y0)
            b0s = helper(b0)
            beta_on_0s = helper(beta_on_0)

        with tf.device(get_protocol().server_1.device_name):
            y1s = helper(y1)
            b1s = helper(b1)
            beta_on_1s = helper(beta_on_1)

    tensors = []
    for y0, y1, b, b0, b1, beta_on_0, beta_on_1 in zip(y0s, y1s, bs, b0s, b1s, beta_on_0s, beta_on_1s):
        y = PrivateTensor(y0, y1)
        y_masked = MaskedPrivateTensor(y, b, b0, b1, beta_on_0, beta_on_1)
        tensors.append(y_masked)

    return tensors


def scale(x, k, apply_encoding=None):
    assert type(x) in [PrivateTensor], type(x)
    assert type(k) in [int, float], type(k)

    x0, x1 = x.unwrapped
    assert type(x0) in [Int100Tensor], type(x0)
    assert type(x1) in [Int100Tensor], type(x1)

    if apply_encoding is None:
        # determine automatically
        apply_encoding = type(k) is float

    c = np.array([k])
    if apply_encoding:
        c = encode(c)

    with tf.name_scope('scale'):

        with tf.device(get_protocol().server_0.device_name):
            y0 = x0 * c

        with tf.device(get_protocol().server_1.device_name):
            y1 = x1 * c

    y = PrivateTensor(y0, y1)
    if apply_encoding:
        y = truncate(y)

    return y


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

                with tf.device(get_protocol().server_0.device_name):
                    cached_x0 = [tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(x0, m)]
                    updators.append([tf.assign(var, val) for var, val in zip(cached_x0, x0)])

                with tf.device(get_protocol().server_1.device_name):
                    cached_x1 = [tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(x1, m)]
                    updators.append([tf.assign(var, val) for var, val in zip(cached_x1, x1)])

            # TODO[Morten] wrap PrivateTensor around var.read_value() instead to ensure updated values?
            cached = PrivateTensor(cached_x0, cached_x1)
            _nodes[node_key] = cached

        elif isinstance(x, MaskedPrivateTensor):

            a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped
            cached_x = cache(x.unmasked, initializers, updators)

            with tf.name_scope('cache'):

                with tf.device(get_protocol().crypto_producer.device_name):
                    cached_a = [tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(a, m)]
                    updators.append([tf.assign(var, val) for var, val in zip(cached_a, a)])

                with tf.device(get_protocol().server_0.device_name):
                    cached_a0 = [tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(a0, m)]
                    updators.append([tf.assign(var, val) for var, val in zip(cached_a0, a0)])

                    cached_alpha_on_0 = [tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(alpha_on_0, m)]
                    updators.append([tf.assign(var, val) for var, val in zip(cached_alpha_on_0, alpha_on_0)])

                with tf.device(get_protocol().server_1.device_name):
                    cached_a1 = [tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(a1, m)]
                    updators.append([tf.assign(var, val) for var, val in zip(cached_a1, a1)])

                    cached_alpha_on_1 = [tf.Variable(tf.random_uniform(shape=vi.shape, maxval=mi, dtype=INT_TYPE), dtype=INT_TYPE) for vi, mi in zip(alpha_on_1, m)]
                    updators.append([tf.assign(var, val) for var, val in zip(cached_alpha_on_1, alpha_on_1)])

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


def encode_input(vars_and_values):
    if not isinstance(vars_and_values, list):
        vars_and_values = [vars_and_values]
    result = dict()
    for input_x, X in vars_and_values:
        result.update((input_xi, Xi) for input_xi, Xi in zip(input_x, decompose(encode(X))))
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
        for xi in x:
            assert type(xi) == np.ndarray
        # convert all values to floats; at least for small
        # ints this should give correct results
        x = [xi.astype(float) for xi in x]
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
        x = [tf.cast(xi, dtype=tf.int32) for xi in x]
        return x

    y = tf.py_func(patched_func, [x], tf.double)
    y = pyfunc_hack_postexit(y, shape or x.shape)
    return y


def encode_and_decompose(x, shape=None):
    func = lambda x: decompose(encode(x))
    return pyfunc_hack(func, x, shape=shape)


def recombine_and_decode(x):
    return decode(recombine(x))
