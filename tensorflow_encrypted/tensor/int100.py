from __future__ import absolute_import

from functools import reduce
import math

import numpy as np
import tensorflow as tf
from typing import Union, Optional, List, Any

from .crt import (
    gen_crt_decompose, gen_crt_recombine_lagrange, gen_crt_recombine_explicit,
    gen_crt_add, gen_crt_sub, gen_crt_mul, gen_crt_matmul, gen_crt_mod,
    gen_crt_reduce_sum, gen_crt_cumsum, crt_im2col, crt_matmul_split,
    gen_crt_equal_zero, gen_crt_equal,
    gen_crt_sample_uniform, gen_crt_sample_bounded
)
from .helpers import prod, inverse
from .factory import AbstractFactory, AbstractTensor, AbstractConstant, AbstractVariable, AbstractPlaceholder
from .shared import binarize, conv2d

#
# 32 bit CRT
# - we need this to do matmul as int32 is the only supported type for that
# - tried tf.float64 but didn't work out of the box
# - 10 components for modulus ~100 bits
#

INT_TYPE = tf.int32

m = [1201, 1433, 1217, 1237, 1321, 1103, 1129, 1367, 1093, 1039]
M = prod(m)
BITSIZE = math.ceil(math.log2(M))

# make sure we have room for lazy reductions:
# - 1 multiplication followed by 1024 additions
for mi in m:
    assert 2 * math.log2(mi) + math.log2(1024) < math.log2(INT_TYPE.max)

MATMUL_THRESHOLD = 1024
SUM_THRESHOLD = 2**9

_crt_decompose = gen_crt_decompose(m)
_crt_recombine_lagrange = gen_crt_recombine_lagrange(m)
_crt_recombine_explicit = gen_crt_recombine_explicit(m, INT_TYPE)
_crt_add = gen_crt_add(m)
_crt_reduce_sum = gen_crt_reduce_sum(m)
_crt_cumsum = gen_crt_cumsum(m)
_crt_sub = gen_crt_sub(m)
_crt_mul = gen_crt_mul(m)
_crt_matmul = gen_crt_matmul(m)
_crt_mod = gen_crt_mod(m, INT_TYPE)
_crt_equal_zero = gen_crt_equal_zero(m, INT_TYPE)
_crt_equal = gen_crt_equal(m, INT_TYPE)
_crt_sample_uniform = gen_crt_sample_uniform(m, INT_TYPE)
_crt_sample_bounded = gen_crt_sample_bounded(m, INT_TYPE)

Backing = Union[List[np.ndarray], List[tf.Tensor]]


class Int100Factory(AbstractFactory):

    def zero(self) -> 'Int100Tensor':
        return Int100Tensor([np.array([0])] * len(m))

    def one(self) -> 'Int100Tensor':
        return Int100Tensor([np.array([1])] * len(m))

    def sample_uniform(self, shape: List[int]) -> 'Int100Tensor':
        backing = _crt_sample_uniform(shape)
        return Int100Tensor(backing)

    def sample_bounded(self, shape: List[int], bitlength: int) -> 'Int100Tensor':
        backing = _crt_sample_bounded(shape, bitlength)
        return Int100Tensor(backing)

    def stack(self, xs: List['Int100Tensor'], axis: int = 0) -> 'Int100Tensor':
        assert all(isinstance(x, Int100Tensor) for x in xs)
        backing = [
            tf.stack([x.backing[i] for x in xs], axis=axis)
            for i in range(len(xs[0].backing))
        ]
        return Int100Tensor(backing)

    def concat(self, xs: List['Int100Tensor'], axis: int = 0) -> 'Int100Tensor':
        assert all(isinstance(x, Int100Tensor) for x in xs)
        backing = [
            tf.concat([x.backing[i] for x in xs], axis=axis)
            for i in range(len(xs[0].backing))
        ]
        return Int100Tensor(backing)

    def tensor(self, value) -> 'Int100Tensor':

        if isinstance(value, (tf.Tensor, np.ndarray)):
            return Int100Tensor(_crt_decompose(value))

        if isinstance(value, Int100Tensor):
            return Int100Tensor(value.backing)

        raise TypeError("Don't know how to handle {}", type(value))

    def constant(self, value) -> 'Int100Constant':

        if isinstance(value, (tf.Tensor, np.ndarray)):
            return Int100Constant(_crt_decompose(value))

        if isinstance(value, Int100Tensor):
            return Int100Constant(value.backing)

        raise TypeError("Don't know how to handle {}", type(value))

    def variable(self, initial_value) -> 'Int100Variable':

        if isinstance(initial_value, (tf.Tensor, np.ndarray)):
            return Int100Variable(_crt_decompose(initial_value))

        if isinstance(initial_value, Int100Tensor):
            return Int100Variable(initial_value.backing)

        raise TypeError("Don't know how to handle {}", type(initial_value))

    def placeholder(self, shape: List[int]) -> 'Int100Placeholder':
        return Int100Placeholder(shape)

    @property
    def modulus(self) -> int:
        return M

    @property
    def native_type(self):
        return INT_TYPE


int100factory = Int100Factory()


class Int100Tensor(AbstractTensor):

    modulus = M

    @property
    def factory(self):
        return int100factory

    def __init__(self, backing: Backing) -> None:
        # TODO[Morten] turn any np.ndarray into a tf.Constant to only store tf.Tensors?
        assert type(backing) in [tuple, list], type(backing)
        assert len(backing) == len(m), len(backing)
        self.backing = backing  # type: Union[List[np.ndarray], List[tf.Tensor]]

    def convert_to_tensor(self) -> 'Int100Tensor':
        converted_backing = [
            tf.convert_to_tensor(xi, dtype=self.factory.native_type)
            for xi in self.backing
        ]
        return Int100Tensor(converted_backing)

    def to_native(self) -> Union[tf.Tensor, np.ndarray]:
        return _crt_recombine_explicit(self.backing, 2**32)

    def to_bits(self, factory: Optional[AbstractFactory] = None) -> AbstractTensor:

        factory = factory or self.factory

        with tf.name_scope('to_bits'):

            # we will extract the bits in chunks of 16 as that's reasonable for the explicit CRT
            MAX_CHUNK_BITSIZE = 16
            q, r = BITSIZE // MAX_CHUNK_BITSIZE, BITSIZE % MAX_CHUNK_BITSIZE
            chunk_bitsizes = [MAX_CHUNK_BITSIZE] * q + ([r] if r > 0 else [])

            # extract bits of chunks
            chunks_bits = []  # type: List[INT_TYPE]
            remaining = self
            for chunk_bitsize in chunk_bitsizes:
                # extract chunk from remaining
                chunk = _crt_mod(remaining.backing, 2**chunk_bitsize)
                # extract bits from chunk and save for concatenation later
                chunk_bits = binarize(chunk, chunk_bitsize)
                chunks_bits.append(chunk_bits)
                # perform right shift on remaining
                remaining = (remaining - int100factory.tensor(chunk)) * inverse(2**chunk_bitsize, self.modulus)

            # combine bits of chunks
            bits = tf.concat(chunks_bits, axis=-1)

            return factory.tensor(bits)

    def to_bigint(self) -> np.ndarray:
        return _crt_recombine_lagrange(self.backing)

    def __getitem__(self, slice):
        return Int100Tensor([x[slice] for x in self.backing])

    def __repr__(self) -> str:
        return 'Int100Tensor({})'.format(self.shape)

    @property
    def shape(self) -> List[int]:
        return self.backing[0].shape

    @staticmethod
    def lift(x) -> 'Int100Tensor':

        if isinstance(x, Int100Tensor):
            return x

        if type(x) is int:
            return int100factory.tensor(np.array([x]))

        raise TypeError("Unsupported type {}".format(type(x)))

    def __add__(self, other) -> 'Int100Tensor':
        x, y = Int100Tensor.lift(self), Int100Tensor.lift(other)
        return x.add(y)

    def __radd__(self, other) -> 'Int100Tensor':
        x, y = Int100Tensor.lift(self), Int100Tensor.lift(other)
        return x.add(y)

    def __sub__(self, other) -> 'Int100Tensor':
        x, y = Int100Tensor.lift(self), Int100Tensor.lift(other)
        return x.sub(y)

    def __rsub__(self, other) -> 'Int100Tensor':
        x, y = Int100Tensor.lift(self), Int100Tensor.lift(other)
        return x.sub(y)

    def __mul__(self, other) -> 'Int100Tensor':
        x, y = Int100Tensor.lift(self), Int100Tensor.lift(other)
        return x.mul(y)

    def __rmul__(self, other) -> 'Int100Tensor':
        x, y = Int100Tensor.lift(self), Int100Tensor.lift(other)
        return x.mul(y)

    def __mod__(self, k) -> 'Int100Tensor':
        return self.mod(k)

    def add(self, other) -> 'Int100Tensor':
        x, y = Int100Tensor.lift(self), Int100Tensor.lift(other)
        return Int100Tensor(_crt_add(x.backing, y.backing))

    def sub(self, other) -> 'Int100Tensor':
        x, y = Int100Tensor.lift(self), Int100Tensor.lift(other)
        return Int100Tensor(_crt_sub(x.backing, y.backing))

    def mul(self, other) -> 'Int100Tensor':
        x, y = Int100Tensor.lift(self), Int100Tensor.lift(other)
        return Int100Tensor(_crt_mul(x.backing, y.backing))

    def matmul(self, other) -> 'Int100Tensor':
        x, y = Int100Tensor.lift(self), Int100Tensor.lift(other)

        if x.shape[1] <= MATMUL_THRESHOLD:
            z_backing = _crt_matmul(x.backing, y.backing)

        else:
            split_backing = crt_matmul_split(x.backing, y.backing, MATMUL_THRESHOLD)
            split_products = [_crt_matmul(xi, yi) for xi, yi in split_backing]
            z_backing = reduce(_crt_add, split_products)

        return Int100Tensor(z_backing)

    def mod(self, k: int) -> 'Int100Tensor':
        return Int100Tensor(_crt_decompose(_crt_mod(self.backing, k)))

    def reduce_sum(self, axis=None, keepdims=None) -> 'Int100Tensor':
        backing = _crt_reduce_sum(self.backing, axis, keepdims)
        return Int100Tensor(backing)

    def cumsum(self, axis, exclusive, reverse) -> 'Int100Tensor':
        backing = _crt_cumsum(self.backing, axis=axis, exclusive=exclusive, reverse=reverse)
        return Int100Tensor(backing)

    def equal_zero(self, out_dtype: Optional[AbstractFactory]=None) -> 'Int100Tensor':
        out_dtype = out_dtype or self.factory
        return out_dtype.tensor(_crt_equal_zero(self.backing, out_dtype.native_type))

    def equal(self, other) -> 'Int100Tensor':
        x, y = Int100Tensor.lift(self), Int100Tensor.lift(other)
        out_dtype = x.factory
        return out_dtype.tensor(_crt_equal(x.backing, y.backing, out_dtype.native_type))

    def im2col(self, h_filter, w_filter, padding, strides) -> 'Int100Tensor':
        backing = crt_im2col(self.backing, h_filter, w_filter, padding, strides)
        return Int100Tensor(backing)

    def conv2d(self, other, strides, padding='SAME') -> 'Int100Tensor':
        x, y = Int100Tensor.lift(self), Int100Tensor.lift(other)
        return conv2d(x, y, strides, padding)  # type: ignore

    def transpose(self, perm: Optional[List[int]]=None) -> 'Int100Tensor':
        backing = [tf.transpose(xi, perm=perm) for xi in self.backing]
        return Int100Tensor(backing)

    def strided_slice(self, args: Any, kwargs: Any) -> 'Int100Tensor':
        backing = [tf.strided_slice(xi, *args, **kwargs) for xi in self.backing]
        return Int100Tensor(backing)

    def split(self, num_split: int, axis: int=0) -> List['Int100Tensor']:
        backings = zip(*[tf.split(xi, num_split, axis=axis) for xi in self.backing])
        return [Int100Tensor(backing) for backing in backings]

    def reshape(self, axes: List[int]) -> 'Int100Tensor':
        backing = [tf.reshape(xi, axes) for xi in self.backing]
        return Int100Tensor(backing)

    def expand_dims(self, axis: Optional[int]=None) -> 'Int100Tensor':
        backing = [tf.expand_dims(xi, axis) for xi in self.backing]
        return Int100Tensor(backing)

    def squeeze(self, axis: Optional[List[int]]=None) -> 'Int100Tensor':
        backing = [tf.squeeze(xi, axis=axis) for xi in self.backing]
        return Int100Tensor(backing)

    def negative(self) -> 'Int100Tensor':
        # TODO[Morten] there's probably a more efficient way
        return int100factory.zero() - self

    def truncate(self, amount, base=2):
        factor = base**amount
        factor_inverse = inverse(factor, self.factory.modulus)
        return (self - (self % factor)) * factor_inverse

    def right_shift(self, bitlength):
        return self.truncate(bitlength, 2)


class Int100Constant(Int100Tensor, AbstractConstant):

    def __init__(self, backing: Backing) -> None:
        super(Int100Constant, self).__init__([tf.constant(vi, dtype=INT_TYPE) for vi in backing])

    def __repr__(self) -> str:
        return 'Int100Constant({})'.format(self.shape)


class Int100Placeholder(Int100Tensor, AbstractPlaceholder):

    def __init__(self, shape: List[int]) -> None:
        self.placeholders = [tf.placeholder(INT_TYPE, shape=shape) for _ in m]
        super(Int100Placeholder, self).__init__(self.placeholders)

    def __repr__(self):
        return 'Int100Placeholder({})'.format(self.shape)

    def feed_from_native(self, value):
        assert isinstance(value, np.ndarray), type(value)
        return self.feed_from_same(int100factory.tensor(value))

    def feed_from_same(self, value):
        assert isinstance(value, Int100Tensor), type(value)
        return {
            p: v for p, v in zip(self.placeholders, value.backing)
        }


class Int100Variable(Int100Tensor, AbstractVariable):

    def __init__(self, initial_value: Backing) -> None:
        self.variables = [
            tf.Variable(val, dtype=INT_TYPE, trainable=False)
            for val in initial_value
        ]
        self.initializer = tf.group(*[var.initializer for var in self.variables])
        backing = [
            var.read_value()
            for var in self.variables
        ]
        super(Int100Variable, self).__init__(backing)

    def __repr__(self):
        return 'Int100Variable({})'.format(self.shape)

    def assign_from_native(self, value: np.ndarray):
        assert isinstance(value, np.ndarray), type(value)
        return self.assign_from_same(int100factory.tensor(value))

    def assign_from_same(self, value: Int100Tensor):
        assert isinstance(value, Int100Tensor), type(value)
        return tf.group(*[tf.assign(xi, vi).op for xi, vi in zip(self.variables, value.backing)])
