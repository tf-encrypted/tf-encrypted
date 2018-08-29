from __future__ import absolute_import

import math
import numpy as np
import tensorflow as tf
from typing import Union, Optional, List, Dict, Any

from .crt import (
    gen_crt_decompose, gen_crt_recombine_lagrange, gen_crt_recombine_explicit,
    gen_crt_add, gen_crt_sub, gen_crt_mul, gen_crt_dot, gen_crt_im2col, gen_crt_mod,
    gen_crt_sample_uniform
)
from .helpers import prod, log2
from ..config import run
from typing import Any, List, Tuple


#
# 32 bit CRT
# - we need this to do dot product as int32 is the only supported type for that
# - tried tf.float64 but didn't work out of the box
# - 10 components for modulus ~100 bits
#

INT_TYPE = tf.int32

m = [1201, 1433, 1217, 1237, 1321, 1103, 1129, 1367, 1093, 1039]
M = prod(m)

# make sure we have room for lazy reductions:
# - 1 multiplication followed by 1024 additions
for mi in m:
    assert 2*log2(mi) + log2(1024) < log2(INT_TYPE.max)

_crt_decompose = gen_crt_decompose(m)
_crt_recombine_lagrange = gen_crt_recombine_lagrange(m)
_crt_recombine_explicit = gen_crt_recombine_explicit(m, INT_TYPE)

_crt_add = gen_crt_add(m)
_crt_sub = gen_crt_sub(m)
_crt_mul = gen_crt_mul(m)
_crt_dot = gen_crt_dot(m)
_crt_im2col = gen_crt_im2col(m)
_crt_mod = gen_crt_mod(m, INT_TYPE)

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
    def from_native(value: Union[np.ndarray, tf.Tensor]) -> 'Int100Tensor':
        # TODO[Morten] rename to `from_natural` to highlight that you can feed: int32, int64, bigint
        assert isinstance(value, (np.ndarray, tf.Tensor)), type(value)
        return Int100Tensor(value, None)

    @staticmethod
    def from_decomposed(value: Union[List[np.ndarray], List[tf.Tensor]]) -> 'Int100Tensor':
        assert type(value) in [tuple, list], type(value)
        return Int100Tensor(None, value)

    def eval(self, sess: tf.Session, feed_dict: Dict[Any, Any]={}, tag: Optional[str]=None) -> 'Int100Tensor':
        evaluated_backing: Union[List[np.ndarray], List[tf.Tensor]] = run(sess, self.backing, feed_dict=feed_dict, tag=tag)
        return Int100Tensor.from_decomposed(evaluated_backing)

    def to_int32(self) -> Union[tf.Tensor, np.ndarray]:
        return _crt_recombine_explicit(self.backing, 2**31)

    def to_bigint(self) -> np.ndarray:
        return _crt_recombine_lagrange(self.backing)

    @staticmethod
    def sample_uniform(shape: List[int]) -> 'Int100Tensor':
        return _sample_uniform(shape)

    def __repr__(self) -> str:
        return 'Int100Tensor({})'.format(self.shape)

    @property
    def shape(self) -> List[int]:
        return self.backing[0].shape

    def __add__(self, other: 'Int100Tensor') -> 'Int100Tensor':
        return _add(self, other)

    def __sub__(self, other: 'Int100Tensor') -> 'Int100Tensor':
        return _sub(self, other)

    def __mul__(self, other: 'Int100Tensor') -> 'Int100Tensor':
        return _mul(self, other)

    def dot(self, other: 'Int100Tensor') -> 'Int100Tensor':
        return _dot(self, other)

    def im2col(self, h_filter, w_filter, padding, strides) -> 'Int100Tensor':
        return _im2col(self, h_filter, w_filter, padding, strides)

    def conv2d(self, other, strides, padding='SAME') -> 'Int100Tensor':
        return _conv2d(self, other, strides, padding)

    def __mod__(self, k) -> 'Int100Tensor':
        return _mod(self, k)

    def transpose(self, *axes) -> 'Int100Tensor':
        return _transpose(self, *axes)

    def strided_slice(self, args: Any, kwargs: Any):
        return _strided_slice(self, args, kwargs)

    def reshape(self, *axes) -> 'Int100Tensor':
        return _reshape(self, *axes)


def _lift(x):
    # TODO[Morten] support other types of `x`

    if isinstance(x, Int100Tensor):
        return x

    if type(x) is int:
        return Int100Tensor.from_native(np.array([x]))

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
    backing = [tf.transpose(xi, axes) for xi in x.backing]
    return Int100Tensor.from_decomposed(backing)


def _strided_slice(x: Int100Tensor, args: Any, kwargs: Any):
    assert isinstance(x, Int100Tensor), type(x)
    backing = [tf.strided_slice(xi, *args, **kwargs) for xi in x.backing]
    return Int100Tensor.from_decomposed(backing)


def _reshape(x, *axes):
    assert isinstance(x, Int100Tensor), type(x)
    backing = [tf.reshape(xi, axes) for xi in x.backing]
    return Int100Tensor.from_decomposed(backing)


def stack(x: List[Int100Tensor], axis: int = 0):
    assert all([isinstance(i, Int100Tensor) for i in x])

    backing = []
    for i in range(len(x[0].backing)):
        stacked = [j.backing[i] for j in x]

        backing.append(tf.stack(stacked, axis=axis))

    return Int100Tensor.from_decomposed(backing)


class Int100Constant(Int100Tensor):

    def __init__(self, native_value: np.ndarray, int100_value=None) -> None:
        if int100_value is None:
            int100_value = Int100Tensor.from_native(native_value)

        assert type(int100_value) in [Int100Tensor], type(int100_value)

        backing = [tf.convert_to_tensor(vi, dtype=INT_TYPE) for vi in int100_value.backing]

        super(Int100Constant, self).__init__(None, backing)

    @staticmethod
    def from_native(value: np.ndarray) -> 'Int100Constant':
        assert type(value) in [np.ndarray, tf.Tensor], type(value)
        return Int100Constant(value, None)

    @staticmethod
    def from_int100(value: Int100Tensor) -> 'Int100Constant':
        assert type(value) in [Int100Tensor], type(value)
        return Int100Constant(None, value)

    def __repr__(self) -> str:
        return 'Int100Constant({})'.format(self.shape)


class Int100Placeholder(Int100Tensor):

    def __init__(self, shape):
        placeholders = [tf.placeholder(INT_TYPE, shape=shape) for _ in m]

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

        variables = [tf.Variable(vi, dtype=Int100Tensor.int_type, trainable=False) for vi in int100_initial_value.backing]
        backing = [vi.read_value() for vi in variables]

        super(Int100Variable, self).__init__(None, backing)
        self.variables = variables
        self.initializer = tf.group(*[var.initializer for var in variables])

    @staticmethod
    def from_native(initial_value):
        assert type(initial_value) in [np.ndarray, tf.Tensor], type(initial_value)
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

    ops = [tf.assign(xi, vi).op for xi, vi in zip(variable.variables, decomposed_value.backing)]
    return tf.group(*ops)
