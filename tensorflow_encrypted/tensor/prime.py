from __future__ import absolute_import
from typing import Union, Optional, List, Dict, Any, Tuple
import math

import numpy as np
import tensorflow as tf

from .factory import AbstractFactory, AbstractTensor, AbstractConstant, AbstractVariable, AbstractPlaceholder
from .shared import binarize, im2col


class PrimeTensor(AbstractTensor):

    def __init__(self, value: Union[np.ndarray, tf.Tensor], factory: 'PrimeFactory') -> None:
        self._factory = factory
        self.modulus = factory.modulus
        self.value = value

    def to_native(self) -> Union[tf.Tensor, np.ndarray]:
        return self.value

    def to_bits(self, factory: Optional[AbstractFactory] = None) -> 'PrimeTensor':
        factory = factory or self.factory
        bitsize = math.ceil(math.log2(self.modulus))
        return factory.tensor(binarize(self.value % self.modulus, bitsize))

    def __getitem__(self, slice: Any) -> Union[tf.Tensor, np.ndarray]:
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

    def mod(self, k: int) -> 'PrimeTensor':
        return self.factory.tensor((self.value % k) % self.modulus)

    def transpose(self, perm: Union[List[int], Tuple[int]]) -> 'PrimeTensor':
        return self.factory.tensor(tf.transpose(self.value, perm))

    def strided_slice(self, args: Any, kwargs: Any) -> 'PrimeTensor':
        return self.factory.tensor(tf.strided_slice(self.value, *args, **kwargs))

    def split(self, num_split: int, axis: int=0) -> List['PrimeTensor']:
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

    def equal_zero(self, out_dtype: Optional[AbstractFactory]=None) -> 'PrimeTensor':
        out_dtype = out_dtype or self.factory
        return out_dtype.tensor(tf.cast(tf.equal(self.value, 0), dtype=out_dtype.native_type))

    def cast(self, factory):
        return factory.tensor(self.value)


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

    def __init__(self, value: Union[tf.Tensor, np.ndarray], factory) -> None:
        v = tf.constant(value, dtype=factory.native_type)
        super(PrimeConstant, self).__init__(v, factory)

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
        self.native_type = native_type

    @property
    def modulus(self):
        return self._modulus

    def sample_uniform(self, shape: Union[Tuple[int, ...], tf.TensorShape], minval: Optional[int] = 0) -> PrimeTensor:
        value = tf.random_uniform(shape=shape, dtype=self.native_type, minval=minval, maxval=self.modulus)
        return PrimeTensor(value, self)

    def sample_bounded(self, shape: List[int], bitlength: int) -> PrimeTensor:
        maxval = 2 ** bitlength
        assert self.modulus > maxval
        value = tf.random_uniform(shape=shape, dtype=self.native_type, minval=0, maxval=maxval)
        return PrimeTensor(value, self)

    def stack(self, xs: List[PrimeTensor], axis: int = 0) -> PrimeTensor:
        assert all(isinstance(x, PrimeTensor) for x in xs)
        value = tf.stack([x.value for x in xs], axis=axis)
        return PrimeTensor(value, self)

    def concat(self, xs: List[PrimeTensor], axis: int = 0) -> PrimeTensor:
        assert all(isinstance(x, PrimeTensor) for x in xs)
        value = tf.concat([v.value for v in xs], axis=axis)
        return PrimeTensor(value, self)

    def tensor(self, value) -> PrimeTensor:

        if isinstance(value, (tf.Tensor, np.ndarray)):
            return PrimeTensor(value, self)

        if isinstance(value, PrimeTensor):
            assert value.modulus == self.modulus, "Incompatible modulus: {}, (expected {})".format(value.modulus, self.modulus)
            return PrimeTensor(value.value, self)

        raise TypeError("Don't know how to handle {}".format(type(value)))

    def constant(self, value) -> PrimeConstant:

        if isinstance(value, (tf.Tensor, np.ndarray)):
            return PrimeConstant(value, self)

        if isinstance(value, PrimeTensor):
            assert value.modulus == self.modulus, "Incompatible modulus: {}, (expected {})".format(value.modulus, self.modulus)
            return PrimeConstant(value.value, self)

        raise TypeError("Don't know how to handle {}".format(type(value)))

    def variable(self, initial_value) -> PrimeVariable:

        if isinstance(initial_value, (tf.Tensor, np.ndarray)):
            return PrimeVariable(initial_value, self)

        if isinstance(initial_value, PrimeTensor):
            assert initial_value.modulus == self.modulus, "Incompatible modulus: {}, (expected {})".format(initial_value.modulus, self.modulus)
            return PrimeVariable(initial_value.value, self)

        raise TypeError("Don't know how to handle {}".format(type(initial_value)))

    def placeholder(self, shape: List[int]) -> PrimePlaceholder:
        return PrimePlaceholder(shape, self)
