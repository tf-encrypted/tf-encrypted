from __future__ import absolute_import
from typing import Union, List, Dict, Any, Tuple, Optional

import numpy as np
import tensorflow as tf

from .factory import AbstractFactory, AbstractTensor, AbstractVariable, AbstractConstant, AbstractPlaceholder
from .helpers import inverse
from .shared import binarize, conv2d, im2col
from ..types import Slice, Ellipse


class Int64Factory(AbstractFactory):

    def tensor(self, value) -> 'Int64Tensor':

        if isinstance(value, tf.Tensor):
            if value.dtype is not tf.int64:
                value = tf.cast(value, dtype=tf.int64)
            return Int64Tensor(value)

        if isinstance(value, np.ndarray):
            return Int64Tensor(value)

        if isinstance(value, Int64Tensor):
            return Int64Tensor(value.value)

        raise TypeError("Don't know how to handle {}".format(type(value)))

    def constant(self, value) -> 'Int64Constant':

        if isinstance(value, (tf.Tensor, np.ndarray)):
            return Int64Constant(value)

        if isinstance(value, Int64Tensor):
            return Int64Constant(value.value)

        raise TypeError("Don't know how to handle {}".format(type(value)))

    def variable(self, initial_value) -> 'Int64Variable':

        if isinstance(initial_value, (tf.Tensor, np.ndarray)):
            return Int64Variable(initial_value)

        if isinstance(initial_value, Int64Tensor):
            return Int64Variable(initial_value.value)

        raise TypeError("Don't know how to handle {}".format(type(initial_value)))

    def placeholder(self, shape: List[int]) -> 'Int64Placeholder':
        return Int64Placeholder(shape)

    @property
    def modulus(self) -> int:
        return 2**64

    @property
    def native_type(self):
        return tf.int64

    def sample_uniform(self, shape: List[int]) -> 'Int64Tensor':
        value = tf.random_uniform(shape=shape, dtype=self.native_type, minval=tf.int64.min, maxval=tf.int64.max)
        return Int64Tensor(value)

    def sample_bounded(self, shape: List[int], bitlength: int) -> 'Int64Tensor':
        # TODO[Morten] verify that uses of this work for signed integers
        value = tf.random_uniform(shape=shape, dtype=self.native_type, minval=0, maxval=2**bitlength)
        return Int64Tensor(value)

    def stack(self, xs: List['Int64Tensor'], axis: int = 0) -> 'Int64Tensor':
        assert all(isinstance(x, Int64Tensor) for x in xs)
        value = tf.stack([x.value for x in xs], axis=axis)
        return Int64Tensor(value)

    def concat(self, xs: List['Int64Tensor'], axis: int) -> 'Int64Tensor':
        assert all(isinstance(x, Int64Tensor) for x in xs)
        value = tf.concat([x.value for x in xs], axis=axis)
        return Int64Tensor(value)


int64factory = Int64Factory()


def _lift(x, y) -> Tuple['Int64Tensor', 'Int64Tensor']:

    if isinstance(x, Int64Tensor) and isinstance(y, Int64Tensor):
        return x, y

    if isinstance(x, Int64Tensor) and isinstance(y, int):
        return x, x.factory.tensor(np.array([y]))

    if isinstance(x, int) and isinstance(y, Int64Tensor):
        return y.factory.tensor(np.array([x])), y

    raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))


class Int64Tensor(AbstractTensor):

    def __init__(self, value: Union[np.ndarray, tf.Tensor]) -> None:
        self.value = value

    def to_native(self) -> Union[tf.Tensor, np.ndarray]:
        return self.value

    def to_bits(self, factory: Optional[AbstractFactory] = None) -> AbstractTensor:
        factory = factory or int64factory
        return factory.tensor(binarize(self.value))

    def __repr__(self) -> str:
        return 'Int64Tensor(shape={})'.format(self.shape)

    @property
    def shape(self) -> Union[Tuple[int, ...], tf.TensorShape]:
        return self.value.shape

    @property
    def factory(self) -> AbstractFactory:
        return int64factory

    def __add__(self, other) -> 'Int64Tensor':
        x, y = _lift(self, other)
        return x.add(y)

    def __radd__(self, other) -> 'Int64Tensor':
        x, y = _lift(self, other)
        return x.add(y)

    def __sub__(self, other) -> 'Int64Tensor':
        x, y = _lift(self, other)
        return x.sub(y)

    def __rsub__(self, other) -> 'Int64Tensor':
        x, y = _lift(self, other)
        return x.sub(y)

    def __mul__(self, other) -> 'Int64Tensor':
        x, y = _lift(self, other)
        return x.mul(y)

    def __rmul__(self, other) -> 'Int64Tensor':
        x, y = _lift(self, other)
        return x.mul(y)

    def __mod__(self, k: int) -> 'Int64Tensor':
        return self.mod(k)

    def __neg__(self) -> 'Int64Tensor':
        return self.mul(-1)

    def __getitem__(self, slice: Union[Slice, Ellipse]) -> 'Int64Tensor':
        return int64factory.tensor(self.value[slice])

    def add(self, other: Any) -> 'Int64Tensor':
        x, y = _lift(self, other)
        return int64factory.tensor(x.value + y.value)

    def sub(self, other: Any) -> 'Int64Tensor':
        x, y = _lift(self, other)
        return int64factory.tensor(x.value - y.value)

    def mul(self, other: Any) -> 'Int64Tensor':
        x, y = _lift(self, other)
        return int64factory.tensor(x.value * y.value)

    def matmul(self, other: Any) -> 'Int64Tensor':
        x, y = _lift(self, other)
        return int64factory.tensor(tf.matmul(x.value, y.value))

    def im2col(self, h_filter: int, w_filter: int, padding: str, strides: int) -> 'Int64Tensor':
        return int64factory.tensor(im2col(self.value, h_filter, w_filter, padding, strides))

    def conv2d(self, other: Any, strides: int, padding: str='SAME') -> 'Int64Tensor':
        x, y = _lift(self, other)
        return conv2d(x, y, strides, padding)  # type: ignore

    def mod(self, k: int) -> 'Int64Tensor':
        return int64factory.tensor(self.value % k)

    def transpose(self, perm: Union[List[int], Tuple[int]]) -> 'Int64Tensor':
        return int64factory.tensor(tf.transpose(self.value, perm))

    def strided_slice(self, args: Any, kwargs: Any) -> 'Int64Tensor':
        return int64factory.tensor(tf.strided_slice(self.value, *args, **kwargs))

    def split(self, num_split: int, axis: int=0) -> List['Int64Tensor']:
        values = tf.split(self.value, num_split, axis=axis)
        return [int64factory.tensor(value) for value in values]

    def reshape(self, axes: Union[tf.Tensor, List[int]]) -> 'Int64Tensor':
        return Int64Tensor(tf.reshape(self.value, axes))

    def reduce_sum(self, axis, keepdims=None) -> 'Int64Tensor':
        return Int64Tensor(tf.reduce_sum(self.value, axis, keepdims))

    def cumsum(self, axis, exclusive, reverse) -> 'Int64Tensor':
        return Int64Tensor(tf.cumsum(self.value, axis=axis, exclusive=exclusive, reverse=reverse))

    def equal_zero(self, factory: AbstractFactory=int64factory) -> 'AbstractTensor':
        return factory.tensor(tf.cast(tf.equal(self.value, 0), dtype=factory.native_type))

    def equal(self, other, factory: AbstractFactory=int64factory) -> 'AbstractTensor':
        x, y = _lift(self, other)
        return factory.tensor(tf.cast(tf.equal(x.value, y.value), dtype=factory.native_type))

    def truncate(self, amount, base=2) -> 'Int64Tensor':
        if base == 2:
            return self.right_shift(amount)
        else:
            factor = base**amount
            factor_inverse = inverse(factor, self.factory.modulus)
            return (self - (self % factor)) * factor_inverse

    def right_shift(self, bitlength) -> 'Int64Tensor':
        return Int64Tensor(tf.bitwise.right_shift(self.value, bitlength))

    def expand_dims(self, axis: Optional[int]=None) -> 'Int64Tensor':
        return Int64Tensor(tf.expand_dims(self.value, axis))

    def squeeze(self, axis: Optional[List[int]]=None) -> 'Int64Tensor':
        return Int64Tensor(tf.squeeze(self.value, axis=axis))

    def negative(self) -> 'Int64Tensor':
        return Int64Tensor(tf.negative(self.value))


class Int64Constant(Int64Tensor, AbstractConstant):

    def __init__(self, value: Union[tf.Tensor, np.ndarray]) -> None:
        super(Int64Constant, self).__init__(tf.constant(value, dtype=tf.int64))

    def __repr__(self) -> str:
        return 'int64.Constant(shape={})'.format(self.shape)


class Int64Placeholder(Int64Tensor, AbstractPlaceholder):

    def __init__(self, shape: List[int]) -> None:
        self.placeholder = tf.placeholder(tf.int64, shape=shape)
        super(Int64Placeholder, self).__init__(self.placeholder)

    def __repr__(self) -> str:
        return 'Int64Placeholder(shape={})'.format(self.shape)

    def feed_from_native(self, value: np.ndarray) -> Dict[tf.Tensor, np.ndarray]:
        assert type(value) in [np.ndarray], type(value)
        return self.feed_from_same(int64factory.tensor(value))

    def feed_from_same(self, value: Int64Tensor) -> Dict[tf.Tensor, np.ndarray]:
        assert isinstance(value, Int64Tensor), type(value)
        return {
            self.placeholder: value.value
        }


class Int64Variable(Int64Tensor, AbstractVariable):

    def __init__(self, initial_value: Union[tf.Tensor, np.ndarray]) -> None:
        self.variable = tf.Variable(initial_value, dtype=tf.int64, trainable=False)
        self.initializer = self.variable.initializer
        super(Int64Variable, self).__init__(self.variable.read_value())

    def __repr__(self) -> str:
        return 'Int64Variable(shape={})'.format(self.shape)

    def assign_from_native(self, value: np.ndarray) -> tf.Operation:
        assert type(value) in [np.ndarray], type(value)
        return self.assign_from_same(int64factory.tensor(value))

    def assign_from_same(self, value: Int64Tensor) -> tf.Operation:
        assert isinstance(value, Int64Tensor), type(value)
        return tf.assign(self.variable, value.value).op
