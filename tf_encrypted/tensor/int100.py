from __future__ import absolute_import
from typing import Union, Optional, List, Tuple
from functools import reduce
import abc
import math

import numpy as np
import tensorflow as tf

from .crt import (
    gen_crt_decompose, gen_crt_recombine_lagrange, gen_crt_recombine_explicit,
    gen_crt_add, gen_crt_sub, gen_crt_mul, gen_crt_matmul, gen_crt_mod,
    gen_crt_reduce_sum, gen_crt_cumsum, crt_im2col, crt_matmul_split,
    crt_batch_to_space_nd, crt_space_to_batch_nd, gen_crt_equal_zero,
    gen_crt_equal, gen_crt_sample_uniform, gen_crt_sample_bounded
)
from .helpers import prod, inverse
from .factory import (AbstractFactory, AbstractTensor, AbstractVariable,
                      AbstractConstant, AbstractPlaceholder)
from .shared import binarize, conv2d
from ..operations.secure_random import seed


def crt_factory(INT_TYPE, m):

    M = prod(m)
    BITSIZE = math.ceil(math.log2(M))

    # make sure we have room for lazy reductions:
    # - 1 multiplication followed by 1024 additions
    for mi in m:
        assert 2 * math.log2(mi) + math.log2(1024) < math.log2(INT_TYPE.max)

    MATMUL_THRESHOLD = 1024

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

    class Factory(AbstractFactory):

        def zero(self):
            backing = [tf.constant(0, dtype=INT_TYPE)] * len(m)
            return DenseTensor(backing)

        def one(self):
            backing = [tf.constant(1, dtype=INT_TYPE)] * len(m)
            return DenseTensor(backing)

        def sample_uniform(self,
                           shape,
                           minval: Optional[int] = None,
                           maxval: Optional[int] = None):
            assert minval is None
            assert maxval is None
            return UniformTensor(shape=shape,
                                 seeds=[seed() for _ in m])

        def sample_bounded(self, shape, bitlength: int):
            backing = _crt_sample_bounded(shape, bitlength)
            return DenseTensor(backing)

        def stack(self, xs: list, axis: int = 0):
            assert all(isinstance(x, Tensor) for x in xs)
            backing = [
                tf.stack([x.backing[i] for x in xs], axis=axis)
                for i in range(len(xs[0].backing))
            ]
            return DenseTensor(backing)

        def concat(self, xs: list, axis: int = 0):
            assert all(isinstance(x, Tensor) for x in xs)
            backing = [
                tf.concat([x.backing[i] for x in xs], axis=axis)
                for i in range(len(xs[0].backing))
            ]
            return DenseTensor(backing)

        def tensor(self, value):

            if isinstance(value, tf.Tensor):
                backing = [
                    tf.cast(component, dtype=INT_TYPE)
                    for component in _crt_decompose(value)
                ]
                return DenseTensor(backing)

            if isinstance(value, np.ndarray):
                backing = [
                    tf.convert_to_tensor(component, dtype=INT_TYPE)
                    for component in _crt_decompose(value)
                ]
                return DenseTensor(backing)

            raise TypeError("Don't know how to handle {}", type(value))

        def constant(self, value):

            if isinstance(value, np.ndarray):
                backing = [
                    tf.constant(v, dtype=INT_TYPE)
                    for v in _crt_decompose(value)
                ]
                return Constant(backing)

            raise TypeError("Don't know how to handle {}", type(value))

        def variable(self, initial_value):

            if isinstance(initial_value, (tf.Tensor, np.ndarray)):
                return Variable(_crt_decompose(initial_value))

            if isinstance(initial_value, Tensor):
                return Variable(initial_value.backing)

            raise TypeError("Don't know how to handle {}", type(initial_value))

        def placeholder(self, shape):
            return Placeholder(shape)

        @property
        def modulus(self) -> int:
            return M

        @property
        def native_type(self):
            return INT_TYPE

    FACTORY = Factory()

    def _lift(x, y) -> Tuple['Tensor', 'Tensor']:

        if isinstance(x, Tensor) and isinstance(y, Tensor):
            return x, y

        if isinstance(x, Tensor):

            if isinstance(y, int):
                return x, x.factory.tensor(np.array([y]))

        if isinstance(y, Tensor):

            if isinstance(x, int):
                return y.factory.tensor(np.array([x])), y

        raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))

    class Tensor(AbstractTensor):

        @abc.abstractproperty
        @property
        def backing(self):
            pass

        @abc.abstractproperty
        @property
        def shape(self):
            pass

        modulus = M

        @property
        def factory(self):
            return FACTORY

        def to_native(self) -> Union[tf.Tensor, np.ndarray]:
            return _crt_recombine_explicit(self.backing, 2**32)

        def bits(
            self,
            factory: Optional[AbstractFactory] = None,
            ensure_positive_interpretation: bool = False
        ) -> AbstractTensor:

            factory = factory or self.factory

            with tf.name_scope('to_bits'):

                # we will extract the bits in chunks of 16 as that's reasonable for the explicit CRT
                MAX_CHUNK_BITSIZE = 16
                q, r = BITSIZE // MAX_CHUNK_BITSIZE, BITSIZE % MAX_CHUNK_BITSIZE
                chunk_bitsizes = [MAX_CHUNK_BITSIZE] * q + ([r] if r > 0 else [])
                chunks_modulus = [2**bitsize for bitsize in chunk_bitsizes]

                remaining = self

                if ensure_positive_interpretation:

                    # To get the right bit pattern for negative numbers we need to apply a correction
                    # to the first chunk. Unfortunately, this isn't known until all bits have been
                    # extracted and hence we extract bits both with and without the correction and
                    # select afterwards. Although these two versions could be computed independently
                    # we here combine them into a single tensor to keep the graph smaller.

                    shape = self.shape.as_list()
                    shape_value = [1] + shape
                    shape_correction = [2] + [1] * len(shape)

                    # this means that chunk[0] is uncorrected and chunk[1] is corrected
                    correction_raw = [0, self.modulus % chunks_modulus[0]]
                    correction = tf.constant(correction_raw,
                                             shape=shape_correction,
                                             dtype=self.factory.native_type)

                    remaining = remaining.reshape(shape_value)

                # extract chunks
                chunks = []
                apply_correction = ensure_positive_interpretation
                for chunk_modulus in chunks_modulus:

                    # extract chunk from remaining
                    chunk = _crt_mod(remaining.backing, chunk_modulus)

                    # apply correction only to the first chunk
                    if apply_correction:
                        chunk = (chunk + correction) % chunk_modulus
                        apply_correction = False

                    # save for below
                    chunks.append(chunk)

                    # perform right shift on remaining
                    shifted = (remaining - FACTORY.tensor(chunk))
                    remaining = shifted * inverse(chunk_modulus, self.modulus)

                if ensure_positive_interpretation:
                    # pick between corrected and uncorrected based on MSB
                    msb = chunks[-1][0] >= (chunks_modulus[-1]) // 2
                    chunks = [
                        tf.where(
                            msb,
                            chunk[1],  # corrected
                            chunk[0],  # uncorrected
                        )
                        for chunk in chunks
                    ]

                # extract bits from chunks
                chunks_bits = [
                    binarize(chunk, chunk_bitsize)
                    for chunk, chunk_bitsize in zip(chunks, chunk_bitsizes)
                ]

                # combine bits of chunks
                bits = tf.concat(chunks_bits, axis=-1)

                return factory.tensor(bits)

        def to_bigint(self) -> np.ndarray:
            return _crt_recombine_lagrange(self.backing)

        def __getitem__(self, slice):
            return DenseTensor([x[slice] for x in self.backing])

        def __repr__(self) -> str:
            return 'Tensor({})'.format(self.shape)

        def __add__(self, other):
            x, y = _lift(self, other)
            return x.add(y)

        def __radd__(self, other):
            x, y = _lift(self, other)
            return x.add(y)

        def __sub__(self, other):
            x, y = _lift(self, other)
            return x.sub(y)

        def __rsub__(self, other):
            x, y = _lift(self, other)
            return x.sub(y)

        def __mul__(self, other):
            x, y = _lift(self, other)
            return x.mul(y)

        def __rmul__(self, other):
            x, y = _lift(self, other)
            return x.mul(y)

        def __mod__(self, k: int):
            return self.mod(k)

        def add(self, other):
            x, y = _lift(self, other)
            return DenseTensor(_crt_add(x.backing, y.backing))

        def sub(self, other):
            x, y = _lift(self, other)
            return DenseTensor(_crt_sub(x.backing, y.backing))

        def mul(self, other):
            x, y = _lift(self, other)
            return DenseTensor(_crt_mul(x.backing, y.backing))

        def matmul(self, other):
            x, y = _lift(self, other)

            if x.shape[1] <= MATMUL_THRESHOLD:
                z_backing = _crt_matmul(x.backing, y.backing)

            else:
                split_backing = crt_matmul_split(x.backing, y.backing, MATMUL_THRESHOLD)
                split_products = [_crt_matmul(xi, yi) for xi, yi in split_backing]
                z_backing = reduce(_crt_add, split_products)

            return DenseTensor(z_backing)

        def mod(self, k: int):
            return DenseTensor(_crt_decompose(_crt_mod(self.backing, k)))

        def reduce_sum(self, axis, keepdims=None):
            backing = _crt_reduce_sum(self.backing, axis, keepdims)
            return DenseTensor(backing)

        def cumsum(self, axis, exclusive, reverse):
            backing = _crt_cumsum(self.backing, axis=axis, exclusive=exclusive, reverse=reverse)
            return DenseTensor(backing)

        def equal_zero(self, factory=None):
            factory = factory or self.factory
            return factory.tensor(_crt_equal_zero(self.backing, factory.native_type))

        def equal(self, other, factory=None):
            x, y = _lift(self, other)
            factory = factory or x.factory
            return factory.tensor(_crt_equal(x.backing, y.backing, factory.native_type))

        def im2col(self, h_filter: int, w_filter: int, padding: int, strides: int):
            backing = crt_im2col(self.backing, h_filter, w_filter, padding, strides)
            return DenseTensor(backing)

        def conv2d(self, other, strides: int, padding: str = 'SAME'):
            x, y = _lift(self, other)
            return conv2d(x, y, strides, padding)  # type: ignore

        def batch_to_space_nd(self, block_shape, crops):
            backing = crt_batch_to_space_nd(self.backing, block_shape, crops)
            return DenseTensor(backing)

        def space_to_batch_nd(self, block_shape, paddings):
            backing = crt_space_to_batch_nd(self.backing, block_shape, paddings)
            return DenseTensor(backing)

        def transpose(self, perm):
            backing = [tf.transpose(xi, perm=perm) for xi in self.backing]
            return DenseTensor(backing)

        def strided_slice(self, args, kwargs):
            backing = [tf.strided_slice(xi, *args, **kwargs) for xi in self.backing]
            return DenseTensor(backing)

        def split(self, num_split: int, axis: int = 0):
            backings = zip(*[tf.split(xi, num_split, axis=axis) for xi in self.backing])
            return [DenseTensor(backing) for backing in backings]

        def reshape(self, axes: List[int]):
            backing = [tf.reshape(xi, axes) for xi in self.backing]
            return DenseTensor(backing)

        def expand_dims(self, axis: Optional[int] = None):
            backing = [tf.expand_dims(xi, axis) for xi in self.backing]
            return DenseTensor(backing)

        def squeeze(self, axis: Optional[List[int]] = None):
            backing = [tf.squeeze(xi, axis=axis) for xi in self.backing]
            return DenseTensor(backing)

        def negative(self):
            # TODO[Morten] there's probably a more efficient way
            return FACTORY.zero() - self

        def truncate(self, amount, base=2):
            factor = base**amount
            factor_inverse = inverse(factor, self.factory.modulus)
            return (self - (self % factor)) * factor_inverse

        def right_shift(self, bitlength):
            return self.truncate(bitlength, 2)

        def cast(self, factory):
            # NOTE(Morten) could add more convertion options
            if factory is self.factory:
                return self
            raise ValueError("Don't know how to cast into {}", factory)

    class DenseTensor(Tensor):

        def __init__(self, backing):
            assert isinstance(backing, (tuple, list))
            assert all(isinstance(component, tf.Tensor) for component in backing)
            assert len(backing) == len(m), len(backing)
            self._backing = backing

        @property
        def shape(self):
            return self._backing[0].shape

        @property
        def backing(self):
            return self._backing

    class UniformTensor(Tensor):

        def __init__(self, shape, seeds):
            self._seeds = seeds
            self._shape = shape

        @property
        def shape(self):
            return self._shape

        @property
        def backing(self):
            with tf.name_scope('expand-seed'):
                return _crt_sample_uniform(shape=self._shape,
                                           seeds=self._seeds)

    class Constant(DenseTensor, AbstractConstant):

        def __init__(self, backing) -> None:
            assert isinstance(backing, (list, tuple))
            assert all(isinstance(component, tf.Tensor) for component in backing)
            super(Constant, self).__init__(backing)

        def __repr__(self) -> str:
            return 'Constant({})'.format(self.shape)

    class Placeholder(DenseTensor, AbstractPlaceholder):

        def __init__(self, shape: List[int]) -> None:
            self.placeholders = [tf.placeholder(INT_TYPE, shape=shape) for _ in m]
            super(Placeholder, self).__init__(self.placeholders)

        def __repr__(self):
            return 'Placeholder({})'.format(self.shape)

        def feed(self, value):
            assert isinstance(value, np.ndarray), type(value)
            backing = _crt_decompose(value)
            return {
                p: v for p, v in zip(self.placeholders, backing)
            }

    class Variable(DenseTensor, AbstractVariable):

        def __init__(self, initial_backing) -> None:
            self.variables = [
                tf.Variable(val, dtype=INT_TYPE, trainable=False)
                for val in initial_backing
            ]
            self.initializer = tf.group(*[var.initializer for var in self.variables])
            backing = [
                var.read_value()
                for var in self.variables
            ]
            super(Variable, self).__init__(backing)

        def __repr__(self):
            return 'Variable({})'.format(self.shape)

        def assign_from_native(self, value: np.ndarray):
            assert isinstance(value, np.ndarray), type(value)
            return self.assign_from_same(FACTORY.tensor(value))

        def assign_from_same(self, value: Tensor):
            assert isinstance(value, Tensor), type(value)
            return tf.group(*[tf.assign(xi, vi).op for xi, vi in zip(self.variables, value.backing)])

    return FACTORY


#
# 32 bit CRT
# - we need this to do matmul as int32 is the only supported type for that
# - tried tf.float64 but didn't work out of the box
# - 10 components for modulus ~100 bits
#

int100factory = crt_factory(INT_TYPE=tf.int32,
                            m=[1201, 1433, 1217, 1237, 1321, 1103, 1129, 1367, 1093, 1039])
