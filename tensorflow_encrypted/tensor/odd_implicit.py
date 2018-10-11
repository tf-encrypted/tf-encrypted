from __future__ import absolute_import
from typing import Union, List, Any, Tuple, Type

import numpy as np
import tensorflow as tf

from .factory import AbstractTensor, AbstractFactory


class OddImplicitFactory:

    def __init__(self, native_type: Union[Type['tf.int32'], Type['tf.int64']]) -> None:
        self.native_type = native_type

    def tensor(self, value) -> 'OddImplicitTensor':

        if isinstance(value, (tf.Tensor, np.ndarray)):
            return OddImplicitTensor(value, self)

        if isinstance(value, OddImplicitTensor):
            assert value.factory == self
            return OddImplicitTensor(value.value, self)

        raise TypeError("Don't know how to handle {}".format(type(value)))

    def constant(self, value) -> 'OddImplicitTensor':
        raise NotImplementedError()

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
            minval=self.native_type.min + 1,
            maxval=self.native_type.max)
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
        return self.factory.tensor(self.value[slice])

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

        z = x.value + y.value

        native_type = self.factory.native_type

        # correct for overflow where needed
        z = z + tf.where(
            tf.logical_and(y.value > 0, x.value > native_type.max - y.value),
            tf.ones(z.shape, dtype=native_type),
            tf.zeros(z.shape, dtype=native_type)
        )

        # correct for underflow where needed
        z = z - tf.where(
            tf.logical_and(y.value < 0, x.value < native_type.min - y.value),
            tf.ones(z.shape, dtype=native_type),
            tf.zeros(z.shape, dtype=native_type)
        )

        return OddImplicitTensor(z, self._factory)

    def sub(self, other) -> 'OddImplicitTensor':
        x, y = _lift(self, other)

        z = x.value - y.value

        native_type = self.factory.native_type

        # correct for overflow where needed
        z = z + tf.where(
            tf.logical_and(y.value < 0, x.value > native_type.max + y.value),
            tf.ones(z.shape, dtype=native_type),
            tf.zeros(z.shape, dtype=native_type)
        )

        # correct for underflow where needed
        z = z - tf.where(
            tf.logical_and(y.value > 0, x.value < native_type.min + y.value),
            tf.ones(z.shape, dtype=native_type),
            tf.zeros(z.shape, dtype=native_type)
        )

        return OddImplicitTensor(z, self._factory)

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

    def transpose(self, perm: Union[List[int], Tuple[int]]) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.transpose(self.value, perm), self.factory)

    def strided_slice(self, args: Any, kwargs: Any) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.strided_slice(self.value, *args, **kwargs), self.factory)

    def reshape(self, axes: Union[tf.Tensor, List[int]]) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.reshape(self.value, axes), self.factory)


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
