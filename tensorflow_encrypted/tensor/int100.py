from __future__ import absolute_import

import math
import numpy as np
import tensorflow as tf
from typing import Union, Optional, List, Dict, Any

from .crt import (
    gen_crt_decompose, gen_crt_recombine,
    gen_crt_add, gen_crt_sub, gen_crt_mul, gen_crt_dot, gen_crt_im2col, gen_crt_mod,
    gen_crt_sample_uniform
)
from .helpers import prod, log2
from ..config import run

#
# 32 bit CRT
# - we need this to do dot product as int32 is the only supported type for that
# - tried tf.float64 but didn't work out of the box
# - 10 components for modulus ~100 bits
#

INT_TYPE = tf.int32
FLOAT_TYPE = tf.float32

m = [1201, 1433, 1217, 1237, 1321, 1103, 1129, 1367, 1093, 1039]
M = prod(m)

_lambdas = [
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

# make sure we have room for lazy reductions:
# - 1 multiplication followed by 1024 additions
for mi in m: assert 2*log2(mi) + log2(1024) < log2(INT_TYPE.max)

_crt_decompose = gen_crt_decompose(m)
_crt_recombine = gen_crt_recombine(m, _lambdas)

_crt_add = gen_crt_add(m)
_crt_sub = gen_crt_sub(m)
_crt_mul = gen_crt_mul(m)
_crt_dot = gen_crt_dot(m)
_crt_im2col = gen_crt_im2col(m)
_crt_mod = gen_crt_mod(m, INT_TYPE, FLOAT_TYPE)

_crt_sample_uniform = gen_crt_sample_uniform(m, INT_TYPE)


class Int100Tensor(object):

    modulus = M
    int_type = INT_TYPE

    def __init__(self, native_value: Optional[Union[np.ndarray, tf.Tensor]], decomposed_value: Optional[Union[List[np.ndarray], List[tf.Tensor]]]=None) -> None:
        if decomposed_value is None:
            decomposed_value = _crt_decompose(native_value)

        # TODO[Morten] turn any np.ndarray into a tf.Constant to only store tf.Tensors?
        assert type(decomposed_value) in [tuple, list], type(decomposed_value)

        self.backing = decomposed_value

    @staticmethod
    def from_native(value: Union[np.ndarray, tf.Tensor]) -> Int100Tensor:
        assert isinstance(value, (np.ndarray, tf.Tensor)), type(value)
        return Int100Tensor(value, None)

    @staticmethod
    def from_decomposed(value: Union[List[np.ndarray], List[tf.Tensor]]) -> Int100Tensor:
        assert type(value) in [tuple, list], type(value)
        return Int100Tensor(None, value)

    def eval(self, sess: tf.Session, feed_dict: Dict[Any, Any]={}, tag: Optional[str]=None) -> Int100Tensor:
        evaluated_backing: Union[List[np.ndarray], List[tf.Tensor]] = run(sess, self.backing, feed_dict=feed_dict, tag=tag)
        return Int100Tensor.from_decomposed(evaluated_backing)

    def to_native(self) -> Union[List[np.ndarray], List[tf.Tensor]]:
        return _crt_recombine(self.backing).astype(object)

    @staticmethod
    def sample_uniform(shape):
        return _sample_uniform(shape)

    def __repr__(self):
        return 'Int100Tensor({})'.format(self.to_native())

    @property
    def shape(self):
        return self.backing[0].shape

    def __add__(self, other):
        return _add(self, other)

    def __sub__(self, other):
        return _sub(self, other)

    def __mul__(self, other):
        return _mul(self, other)

    def dot(self, other):
        return _dot(self, other)

    def im2col(self, h_filter, w_filter, padding, strides):
        return _im2col(self, h_filter, w_filter, padding, strides)

    def conv2d(self, other, strides, padding='SAME'):
        return _conv2d(self, other, strides, padding)

    def __mod__(self, k):
        return _mod(self, k)

    def transpose(self, *axes):
        return _transpose(self, *axes)

    def reshape(self, *axes):
        return _reshape(self, *axes)


def _lift(x):
    # TODO[Morten] support other types of `x`

    if isinstance(x, Int100Tensor):
        return x

    if type(x) is int:
        return Int100Tensor.from_native(np.ndarray([x]))

    raise TypeError("Unsupported type {}".format(type(x)))


def _add(x, y):
    x, y = _lift(x), _lift(y)
    z_backing = _crt_add(x.backing, y.backing)
    return Int100Tensor.from_decomposed(z_backing)


def _sub(x, y):
    x, y = _lift(x), _lift(y)
    z_backing = _crt_sub(x.backing, y.backing)
    return Int100Tensor.from_decomposed(z_backing)


def _mul(x, y):
    x, y = _lift(x), _lift(y)
    z_backing = _crt_mul(x.backing, y.backing)
    return Int100Tensor.from_decomposed(z_backing)


def _dot(x, y):
    x, y = _lift(x), _lift(y)
    z_backing = _crt_dot(x.backing, y.backing)
    return Int100Tensor.from_decomposed(z_backing)


def _im2col(x, h_filter, w_filter, padding, strides):
    assert isinstance(x, Int100Tensor), type(x)
    backing = _crt_im2col(x.backing, h_filter, w_filter, padding, strides)
    return Int100Tensor.from_decomposed(backing)


def _conv2d(x, y, strides, padding):
    assert isinstance(x, Int100Tensor), type(x)
    assert isinstance(y, Int100Tensor), type(y)

    h_filter, w_filter, d_filters, n_filters = map(int, y.shape)
    n_x, d_x, h_x, w_x = map(int, x.shape)
    if padding == "SAME":
        h_out = int(math.ceil(float(h_x) / float(strides)))
        w_out = int(math.ceil(float(w_x) / float(strides)))
    if padding == "VALID":
        h_out = int(math.ceil(float(h_x - h_filter + 1) / float(strides)))
        w_out = int(math.ceil(float(w_x - w_filter + 1) / float(strides)))

    X_col = x.im2col(h_filter, w_filter, padding, strides)
    W_col = y.transpose(3, 2, 0, 1).reshape(int(n_filters), -1)
    out = W_col.dot(X_col)

    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    return out


def _mod(x, k):
    y_backing = _crt_mod(x.backing, k)
    return Int100Tensor.from_decomposed(y_backing)


def _sample_uniform(shape):
    backing = _crt_sample_uniform(shape)
    return Int100Tensor.from_decomposed(backing)


def _transpose(x, *axes):
    assert isinstance(x, Int100Tensor), type(x)
    backing = [ tf.transpose(xi, axes) for xi in x.backing ]
    return Int100Tensor.from_decomposed(backing)


def _reshape(x, *axes):
    assert isinstance(x, Int100Tensor), type(x)
    backing = [ tf.reshape(xi, axes) for xi in x.backing ]
    return Int100Tensor.from_decomposed(backing)


class Int100Constant(Int100Tensor):

    def __init__(self, native_value: np.ndarry, int100_value=None) -> None:
        if int100_value is None:
            int100_value = Int100Tensor.from_native(native_value)

        assert type(int100_value) in [Int100Tensor], type(int100_value)

        backing = [tf.constant(vi, dtype=Int100Tensor.int_type) for vi in int100_value.backing]

        super(Int100Constant, self).__init__(None, backing)

    @staticmethod
    def from_native(value: np.ndarray) -> Int100Constant:
        assert type(value) in [np.ndarray], type(value)
        return Int100Constant(value, None)

    @staticmethod
    def from_int100(value: Int100Tensor) -> Int100Constant:
        assert type(value) in [Int100Tensor], type(value)
        return Int100Constant(None, value)

    def __repr__(self) -> str:
        return 'Int100Constant({})'.format(self.shape)

class Int100Placeholder(Int100Tensor):

    def __init__(self, shape):
        placeholders = [ tf.placeholder(INT_TYPE, shape=shape) for _ in m ]

        super(Int100Placeholder, self).__init__(None, placeholders)
        self.placeholders = placeholders

    def feed_from_native(self, value):
        assert type(value) in [np.ndarray], type(value)
        return _feed(self, value, None)

    def feed_from_int100(self, value):
        assert type(value) in [Int100Tensor], type(value)
        return _feed(self, None, value)

    def __repr__(self):
        return 'Int100Placeholder({})'.format(self.shape)

def _feed(placeholder, native_value, int100_value=None):
    if int100_value is None:
        int100_value = Int100Tensor.from_native(native_value)

    assert type(placeholder) is Int100Placeholder, type(placeholder)
    assert type(int100_value) is Int100Tensor, type(int100_value)

    return {
        p: v for p, v in zip(placeholder.placeholders, int100_value.backing)
    }

class Int100Variable(Int100Tensor):

    def __init__(self, native_initial_value, int100_initial_value=None):
        if int100_initial_value is None:
            int100_initial_value = Int100Tensor.from_native(native_initial_value)

        assert type(int100_initial_value) in [Int100Tensor], type(int100_initial_value)

        variables = [ tf.Variable(vi, dtype=Int100Tensor.int_type) for vi in int100_initial_value.backing ]
        backing   = [ vi.read_value() for vi in variables ]

        super(Int100Variable, self).__init__(None, backing)
        self.variables = variables
        self.initializer = tf.group(*[var.initializer for var in variables])

    @staticmethod
    def from_native(initial_value):
        assert type(initial_value) in [np.ndarray], type(initial_value)
        return Int100Variable(initial_value, None)

    @staticmethod
    def from_int100(initial_value):
        assert type(initial_value) in [Int100Tensor], type(initial_value)
        return Int100Variable(None, initial_value)

    def __repr__(self):
        return 'Int100Variable({})'.format(self.shape)

    def assign_from_native(self, value):
        assert type(value) in [np.ndarray], type(value)
        return _assign(self, value, None)

    def assign_from_int100(self, value):
        assert isinstance(value, Int100Tensor), type(value)
        return _assign(self, None, value)

def _assign(variable, native_value, decomposed_value=None):
    if decomposed_value is None:
        decomposed_value = Int100Tensor.from_native(native_value)

    assert type(variable) in (Int100Variable,), type(variable)
    assert isinstance(decomposed_value, (Int100Tensor,)), type(decomposed_value)

    ops = [ tf.assign(xi, vi).op for xi, vi in zip(variable.variables, decomposed_value.backing) ]
    return tf.group(*ops)
