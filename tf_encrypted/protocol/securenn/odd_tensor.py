from __future__ import absolute_import

from typing import Union, List, Any, Tuple, Type
import math

import numpy as np
import tensorflow as tf

from ...tensor.factory import AbstractFactory, AbstractTensor, AbstractConstant
from ...tensor.shared import binarize


class OddImplicitFactory:

    def __init__(self, native_type: Union[Type['tf.int32'], Type['tf.int64']]) -> None:
        self.native_type = native_type

    def tensor(self, value) -> 'OddImplicitTensor':

        if isinstance(value, (tf.Tensor, np.ndarray)):
            return OddImplicitTensor(value, self)

        if isinstance(value, OddImplicitTensor):
            assert value.factory == self
            raw = value.value
            # map all -1 values to zero since the former value is out of bounds in this ring
            raw = tf.where(raw == -1, tf.zeros(shape=raw.shape, dtype=self.native_type), raw)
            return OddImplicitTensor(raw, self)

        raise TypeError("Don't know how to handle {}".format(type(value)))

    def constant(self, value) -> 'OddImplicitTensor':

        if isinstance(value, (tf.Tensor, np.ndarray)):
            return OddImplicitConstant(value, self)

        if isinstance(value, OddImplicitTensor):
            assert value.factory == self
            return OddImplicitConstant(value.value, self)

        raise TypeError("Don't know how to handle {}".format(type(value)))

    def variable(self, initial_value) -> 'OddImplicitTensor':
        raise NotImplementedError()

    def placeholder(self, shape) -> 'OddImplicitTensor':
        raise NotImplementedError()

    @property
    def modulus(self):

        if self.native_type is tf.int32:
            return 2**32 - 1

        if self.native_type is tf.int64:
            return 2**64 - 1

        raise NotImplementedError("Don't know how to handle {}".format(self.native_type))

    def sample_uniform(self, shape: Union[Tuple[int, ...], tf.TensorShape]) -> 'OddImplicitTensor':
        value = tf.random_uniform(
            shape=shape,
            dtype=self.native_type,
            minval=self.native_type.min,
            maxval=self.native_type.max)
        # map all -1 values to zero since the former value is out of bounds in this ring
        # TODO[Morten] introducing bias; fix once secure randomness is in place
        value = tf.where(value == -1, tf.zeros(shape=value.shape, dtype=self.native_type), value)
        return OddImplicitTensor(value, self)

    def sample_bounded(self, shape: List[int], bitlength: int) -> 'OddImplicitTensor':
        raise NotImplementedError()

    def stack(self, xs: List['OddImplicitTensor'], axis: int = 0) -> 'OddImplicitTensor':
        assert all(isinstance(x, OddImplicitTensor) for x in xs)
        value = tf.stack([x.value for x in xs], axis=axis)
        return OddImplicitTensor(value, self)

    def concat(self, xs: List['OddImplicitTensor'], axis: int) -> 'OddImplicitTensor':
        assert all(isinstance(x, OddImplicitTensor) for x in xs)
        value = tf.concat([x.value for x in xs], axis=axis)
        return OddImplicitTensor(value, self)


oddInt32factory = OddImplicitFactory(tf.int32)
oddInt64factory = OddImplicitFactory(tf.int64)


class OddImplicitTensor(AbstractTensor):

    def __init__(self, value: Union[np.ndarray, tf.Tensor], factory: OddImplicitFactory) -> None:
        self._factory = factory
        self.value = value

    def __repr__(self) -> str:
        return 'OddImplicitTensor(shape={}, native_type={})'.format(self.shape, self._factory.native_type)

    def __getitem__(self, slice: Any) -> Union[tf.Tensor, np.ndarray]:
        return OddImplicitTensor(self.value[slice], self.factory)

    @property
    def shape(self) -> Union[Tuple[int, ...], tf.TensorShape]:
        return self.value.shape

    @property
    def factory(self) -> AbstractFactory:
        return self._factory

    def __add__(self, other) -> 'OddImplicitTensor':
        return self.add(other)

    def __sub__(self, other) -> 'OddImplicitTensor':
        return self.sub(other)

    def __mul__(self, other) -> 'OddImplicitTensor':
        return self.mul(other)

    def __mod__(self, k: int) -> 'OddImplicitTensor':
        return self.mod(k)

    def add(self, other) -> 'OddImplicitTensor':
        x, y = _lift(self, other)
        bitlength = math.ceil(math.log2(self.factory.modulus))

        z = x.value + y.value

        # We want to compute `pos(x) + pos(y) >= m - 1` for correction purposes
        # which, since `m - 1 == 1` for signed integers, can be rewritten as
        #  -> `pos(x) >= m - 1 - pos(y)`
        #  -> `m - 1 - pos(y) - 1 < pos(x)`
        #  -> `-1 - pos(y) - 1 < pos(x)`
        wrapped_around = _lessthan_as_unsigned(-2 - y.value, x.value, bitlength)
        z += wrapped_around

        return OddImplicitTensor(z, self.factory)

    def sub(self, other) -> 'OddImplicitTensor':
        x, y = _lift(self, other)
        bitlength = math.ceil(math.log2(self.factory.modulus))

        z = x.value - y.value

        # We want to compute `pos(x) - pos(y) < 0` for correction purposes
        # which can be rewritten as
        #  -> `pos(x) < pos(y)`
        wrapped_around = _lessthan_as_unsigned(x.value, y.value, bitlength)
        z -= wrapped_around

        return OddImplicitTensor(z, self.factory)

    def mul(self, other) -> 'OddImplicitTensor':
        raise NotImplementedError()

    def matmul(self, other) -> 'OddImplicitTensor':
        raise NotImplementedError()

    def im2col(self, h_filter, w_filter, padding, strides) -> 'OddImplicitTensor':
        raise NotImplementedError()

    def conv2d(self, other, strides, padding='SAME') -> 'OddImplicitTensor':
        raise NotImplementedError()

    def mod(self, k: int) -> 'OddImplicitTensor':
        raise NotImplementedError()

    def bits(self, factory=None) -> AbstractTensor:
        factory = factory or self.factory
        return factory.tensor(binarize(self.value))

    def transpose(self, perm: Union[List[int], Tuple[int]]) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.transpose(self.value, perm), self.factory)

    def strided_slice(self, args: Any, kwargs: Any) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.strided_slice(self.value, *args, **kwargs), self.factory)

    def reshape(self, axes: Union[tf.Tensor, List[int]]) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.reshape(self.value, axes), self.factory)

    def cast(self, dtype):
        return dtype.tensor(self.value)


class OddImplicitConstant(OddImplicitTensor, AbstractConstant):

    def __init__(self, value, factory) -> None:
        v = tf.constant(value, dtype=tf.int64)
        super(OddImplicitConstant, self).__init__(v, factory)

    def __repr__(self) -> str:
        return 'OddImplicitConstant(shape={})'.format(self.shape)


def _lift(x, y) -> Tuple[OddImplicitTensor, OddImplicitTensor]:

    if isinstance(x, OddImplicitTensor) and isinstance(y, OddImplicitTensor):
        assert x.factory == y.factory, "Incompatible data types: {} and {}".format(x.factory, y.factory)
        return x, y

    if isinstance(x, OddImplicitTensor):

        if isinstance(y, int):
            return x, x.factory.tensor(np.array([y]))

        if isinstance(y, np.ndarray):
            return x, x.factory.tensor(y)

    if isinstance(y, OddImplicitTensor):

        if isinstance(x, int):
            return y.factory.tensor(np.array([x])), y

        if isinstance(x, np.ndarray):
            return y.factory.tensor(x), y

    raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))


def _lessthan_as_unsigned(x, y, bitlength):
    # Performs comparison `x < y` on signed integers *as if* they were unsigned, e.g. `1 < -1`.
    # Taken from Section 2-12, page 23, of [Hacker's Delight](https://www.hackersdelight.org/).
    not_x = tf.bitwise.invert(x)
    lhs = tf.bitwise.bitwise_and(not_x, y)
    rhs = tf.bitwise.bitwise_and(tf.bitwise.bitwise_or(not_x, y), x - y)
    z = tf.bitwise.right_shift(tf.bitwise.bitwise_or(lhs, rhs), bitlength - 1)
    # turn 0/-1 into 0/1 before returning
    return tf.bitwise.bitwise_and(z, tf.ones(shape=z.shape, dtype=z.dtype))
