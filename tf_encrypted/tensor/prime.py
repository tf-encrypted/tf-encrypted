from __future__ import absolute_import
from typing import Union, Optional, List, Dict, Any, Tuple
import math

import numpy as np
import tensorflow as tf

from .factory import (AbstractFactory, AbstractTensor, AbstractVariable,
                      AbstractConstant, AbstractPlaceholder)
from .shared import binarize, im2col
from ..operations.secure_random import random_uniform


class PrimeTensor(AbstractTensor):

    def __init__(self, value, factory) -> None:
        assert isinstance(value, tf.Tensor)
        self._factory = factory
        self.modulus = factory.modulus
        self.value = value

    def to_native(self) -> tf.Tensor:
        return self.value

    def bits(self, dtype: Optional[AbstractFactory] = None):
        dtype = dtype or self.factory
        bitsize = math.ceil(math.log2(self.modulus))
        return dtype.tensor(binarize(self.value % self.modulus, bitsize))

    def __getitem__(self, slice):
        return self.factory.tensor(self.value[slice])

    def __repr__(self) -> str:
        return 'PrimeTensor(shape={}, modulus={})'.format(self.shape, self.modulus)

    @property
    def shape(self) -> Union[Tuple[int, ...], tf.TensorShape]:
        return self.value.shape

    @property
    def factory(self) -> AbstractFactory:
        return self._factory

    def __add__(self, other) -> 'PrimeTensor':
        return self.add(other)

    def __sub__(self, other) -> 'PrimeTensor':
        return self.sub(other)

    def __mul__(self, other) -> 'PrimeTensor':
        return self.mul(other)

    def __mod__(self, k: int) -> 'PrimeTensor':
        return self.mod(k)

    def add(self, other) -> 'PrimeTensor':
        x, y = _lift(self, other)
        return self.factory.tensor((x.value + y.value) % self.modulus)

    def sub(self, other) -> 'PrimeTensor':
        x, y = _lift(self, other)
        return self.factory.tensor((x.value - y.value) % self.modulus)

    def mul(self, other) -> 'PrimeTensor':
        x, y = _lift(self, other)
        return self.factory.tensor((x.value * y.value) % self.modulus)

    def negative(self) -> 'PrimeTensor':
        return self.mul(-1)

    def matmul(self, other) -> 'PrimeTensor':
        x, y = _lift(self, other)
        return self.factory.tensor(tf.matmul(x.value, y.value) % self.modulus)

    def im2col(self, h_filter, w_filter, padding, strides) -> 'PrimeTensor':
        return self.factory.tensor(im2col(self.value, h_filter, w_filter, padding, strides))

    def conv2d(self, other, strides, padding='SAME') -> 'PrimeTensor':
        raise NotImplementedError()

    def batch_to_space_nd(self, block_shape, crops):
        backing = tf.batch_to_space_nd(self.value, block_shape, crops)
        return self.factory.tensor(backing)

    def space_to_batch_nd(self, block_shape, paddings):
        backing = tf.space_to_batch_nd(self.value, block_shape, paddings)
        return self.factory.tensor(backing)

    def mod(self, k: int) -> 'PrimeTensor':
        return self.factory.tensor((self.value % k) % self.modulus)

    def transpose(self, perm: Union[List[int], Tuple[int]]) -> 'PrimeTensor':
        return self.factory.tensor(tf.transpose(self.value, perm))

    def strided_slice(self, args: Any, kwargs: Any) -> 'PrimeTensor':
        return self.factory.tensor(tf.strided_slice(self.value, *args, **kwargs))

    def split(self, num_split: int, axis: int = 0) -> List['PrimeTensor']:
        values = tf.split(self.value, num_split, axis=axis)
        return [self.factory.tensor(value) for value in values]

    def reshape(self, axes: Union[tf.Tensor, List[int]]) -> 'PrimeTensor':
        return self.factory.tensor(tf.reshape(self.value, axes))

    def expand_dims(self, axis: int) -> 'PrimeTensor':
        return self.factory.tensor(tf.expand_dims(self.value, axis))

    def reduce_sum(self, axis, keepdims) -> 'PrimeTensor':
        return self.factory.tensor(tf.reduce_sum(self.value, axis, keepdims) % self.modulus)

    def sum(self, axis, keepdims) -> 'PrimeTensor':
        return self.reduce_sum(axis, keepdims)

    def cumsum(self, axis, exclusive, reverse) -> 'PrimeTensor':
        return self.factory.tensor(
            tf.cumsum(self.value, axis=axis, exclusive=exclusive, reverse=reverse) % self.modulus
        )

    def equal_zero(self, dtype: Optional[AbstractFactory] = None) -> 'PrimeTensor':
        dtype = dtype or self.factory
        return dtype.tensor(tf.cast(tf.equal(self.value, 0), dtype=dtype.native_type))

    def cast(self, dtype):
        assert dtype.native_type == self.factory.native_type
        return dtype.tensor(self.value)


def _lift(x, y) -> Tuple[PrimeTensor, PrimeTensor]:

    if isinstance(x, PrimeTensor) and isinstance(y, PrimeTensor):
        assert x.modulus == y.modulus, "Incompatible moduli: {} and {}".format(x.modulus, y.modulus)
        return x, y

    if isinstance(x, PrimeTensor) and isinstance(y, int):
        return x, x.factory.tensor(np.array([y]))

    if isinstance(x, int) and isinstance(y, PrimeTensor):
        return y.factory.tensor(np.array([x])), y

    raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))


class PrimeConstant(PrimeTensor, AbstractConstant):

    def __init__(self, constant: tf.Tensor, factory) -> None:
        assert isinstance(constant, tf.Tensor)
        super(PrimeConstant, self).__init__(constant, factory)

    def __repr__(self) -> str:
        return 'PrimeConstant({})'.format(self.shape)


class PrimePlaceholder(PrimeTensor, AbstractPlaceholder):

    def __init__(self, shape: List[int], factory) -> None:
        placeholder = tf.placeholder(factory.native_type, shape=shape)
        super(PrimePlaceholder, self).__init__(placeholder, factory)
        self.placeholder = placeholder

    def __repr__(self) -> str:
        return 'PrimePlaceholder({})'.format(self.shape)

    def feed_from_native(self, value: np.ndarray) -> Dict[tf.Tensor, np.ndarray]:
        assert isinstance(value, np.ndarray), type(value)
        return self.feed_from_same(self.factory.tensor(value))

    def feed_from_same(self, value: PrimeTensor) -> Dict[tf.Tensor, np.ndarray]:
        assert isinstance(value, PrimeTensor), type(value)
        return {
            self.placeholder: value.value
        }


class PrimeVariable(PrimeTensor, AbstractVariable):

    def __init__(self, initial_value: Union[tf.Tensor, np.ndarray], factory) -> None:
        self.variable = tf.Variable(initial_value, dtype=factory.native_type, trainable=False)
        self.initializer = self.variable.initializer
        super(PrimeVariable, self).__init__(self.variable.read_value(), factory)

    def __repr__(self) -> str:
        return 'PrimeVariable({})'.format(self.shape)

    def assign_from_native(self, value: np.ndarray) -> tf.Operation:
        assert isinstance(value, np.ndarray), type(value)
        return self.assign_from_same(self.factory.tensor(value))

    def assign_from_same(self, value: PrimeTensor) -> tf.Operation:
        assert isinstance(value, (PrimeTensor,)), type(value)
        return tf.assign(self.variable, value.value).op


class PrimeFactory(AbstractFactory):

    def __init__(self, modulus, native_type=tf.int32):
        self._modulus = modulus
        self._native_type = native_type

    @property
    def modulus(self):
        return self._modulus

    @property
    def native_type(self):
        return self._native_type

    def sample_uniform(self,
                       shape,
                       minval: Optional[int] = None,
                       maxval: Optional[int] = None):
        minval = minval or 0
        maxval = maxval or self.modulus
        value = random_uniform(shape=shape,
                               dtype=self.native_type,
                               minval=minval,
                               maxval=maxval)
        return PrimeTensor(value, self)

    def sample_bounded(self, shape, bitlength):
        maxval = 2 ** bitlength
        assert self.modulus > maxval
        value = random_uniform(shape=shape, dtype=self.native_type, minval=0, maxval=maxval)
        return PrimeTensor(value, self)

    def sample_bits(self, shape):
        value = random_uniform(shape=shape, dtype=self.native_type, minval=0, maxval=2)
        return PrimeTensor(value, self)

    def stack(self, xs: list, axis: int = 0):
        assert all(isinstance(x, PrimeTensor) for x in xs)
        value = tf.stack([x.value for x in xs], axis=axis)
        return PrimeTensor(value, self)

    def concat(self, xs: list, axis: int = 0):
        assert all(isinstance(x, PrimeTensor) for x in xs)
        value = tf.concat([v.value for v in xs], axis=axis)
        return PrimeTensor(value, self)

    def tensor(self, value) -> PrimeTensor:

        if isinstance(value, tf.Tensor):
            if value.dtype is not self.native_type:
                value = tf.cast(value, dtype=self.native_type)
            return PrimeTensor(value, self)

        if isinstance(value, np.ndarray):
            value = tf.convert_to_tensor(value, dtype=self.native_type)
            return PrimeTensor(value, self)

        raise TypeError("Don't know how to handle {}".format(type(value)))

    def constant(self, value) -> PrimeConstant:

        if isinstance(value, np.ndarray):
            constant = tf.constant(value, dtype=self.native_type)
            return PrimeConstant(constant, self)

        raise TypeError("Don't know how to handle {}".format(type(value)))

    def variable(self, initial_value) -> PrimeVariable:

        if isinstance(initial_value, (tf.Tensor, np.ndarray)):
            return PrimeVariable(initial_value, self)

        if isinstance(initial_value, PrimeTensor):
            err = "Incompatible modulus: {}, (expected {})".format(initial_value.modulus,
                                                                   self.modulus)
            assert initial_value.modulus == self.modulus, err
            return PrimeVariable(initial_value.value, self)

        raise TypeError("Don't know how to handle {}".format(type(initial_value)))

    def placeholder(self, shape: List[int]) -> PrimePlaceholder:
        return PrimePlaceholder(shape, self)
