from __future__ import absolute_import
from typing import Union, List, Dict, Any, Tuple, Optional

import numpy as np
import tensorflow as tf

from .factory import AbstractFactory, AbstractTensor, AbstractVariable, AbstractConstant, AbstractPlaceholder
from .shared import binarize, conv2d, im2col
from ..types import Slice, Ellipse


class Int32Factory(AbstractFactory):

    def tensor(self, value) -> 'Int32Tensor':

        if isinstance(value, (tf.Tensor, np.ndarray)):
            return Int32Tensor(value)

        if isinstance(value, Int32Tensor):
            return Int32Tensor(value.value)

        raise TypeError("Don't know how to handle {}".format(type(value)))

    def constant(self, value) -> 'Int32Constant':

        if isinstance(value, (tf.Tensor, np.ndarray)):
            return Int32Constant(value)

        if isinstance(value, Int32Tensor):
            return Int32Constant(value.value)

        raise TypeError("Don't know how to handle {}".format(type(value)))

    def variable(self, initial_value) -> 'Int32Variable':

        if isinstance(initial_value, (tf.Tensor, np.ndarray)):
            return Int32Variable(initial_value)

        if isinstance(initial_value, Int32Tensor):
            return Int32Variable(initial_value.value)

        raise TypeError("Don't know how to handle {}".format(type(initial_value)))

    def placeholder(self, shape: List[int]) -> 'Int32Placeholder':
        return Int32Placeholder(shape)

    @property
    def modulus(self) -> int:
        return 2**32

    @property
    def native_type(self):
        return tf.int32

    def sample_uniform(self, shape: List[int]) -> 'Int32Tensor':
        value = tf.random_uniform(shape=shape, dtype=self.native_type, minval=self.native_type.min, maxval=self.native_type.max)
        return Int32Tensor(value)

    def stack(self, xs: List['Int32Tensor'], axis: int = 0) -> 'Int32Tensor':
        assert all(isinstance(x, Int32Tensor) for x in xs)
        value = tf.stack([x.value for x in xs], axis=axis)
        return Int32Tensor(value)

    def concat(self, xs: List['Int32Tensor'], axis: int) -> 'Int32Tensor':
        assert all(isinstance(x, Int32Tensor) for x in xs)
        value = tf.concat([x.value for x in xs], axis=axis)
        return Int32Tensor(value)


int32factory = Int32Factory()


def _lift(x, y) -> Tuple['Int32Tensor', 'Int32Tensor']:

    if isinstance(x, Int32Tensor) and isinstance(y, Int32Tensor):
        return x, y

    if isinstance(x, Int32Tensor) and isinstance(y, int):
        return x, x.factory.tensor(np.array([y]))

    if isinstance(x, int) and isinstance(y, Int32Tensor):
        return y.factory.tensor(np.array([x])), y

    raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))


class Int32Tensor(AbstractTensor):

    def __init__(self, value: Union[np.ndarray, tf.Tensor]) -> None:
        self.value = value

    def to_native(self) -> Union[tf.Tensor, np.ndarray]:
        return self.value

    def to_bits(self, factory: Optional[AbstractFactory] = None) -> AbstractTensor:
        factory = factory or int32factory
        return factory.tensor(binarize(self.value))

    def __repr__(self) -> str:
        return 'Int32Tensor(shape={})'.format(self.shape)

    @property
    def shape(self) -> Union[Tuple[int, ...], tf.TensorShape]:
        return self.value.shape

    @property
    def factory(self) -> AbstractFactory:
        return int32factory

    def __add__(self, other: Any) -> 'Int32Tensor':
        return self.add(other)

    def __sub__(self, other: Any) -> 'Int32Tensor':
        return self.sub(other)

    def __mul__(self, other: Any) -> 'Int32Tensor':
        return self.mul(other)

    def __mod__(self, k: int) -> 'Int32Tensor':
        return self.mod(k)

    def __neg__(self) -> 'Int32Tensor':
        return self.mul(-1)

    def __getitem__(self, slice: Union[Slice, Ellipse]) -> 'Int32Tensor':
        return int32factory.tensor(self.value[slice])

    def add(self, other: Any) -> 'Int32Tensor':
        x, y = _lift(self, other)
        return int32factory.tensor(x.value + y.value)

    def sub(self, other: Any) -> 'Int32Tensor':
        x, y = _lift(self, other)
        return int32factory.tensor(x.value - y.value)

    def mul(self, other: Any) -> 'Int32Tensor':
        x, y = _lift(self, other)
        return int32factory.tensor(x.value * y.value)

    def matmul(self, other: Any) -> 'Int32Tensor':
        x, y = _lift(self, other)
        return int32factory.tensor(tf.matmul(x.value, y.value))

    def im2col(self, h_filter: int, w_filter: int, padding: str, strides: int) -> 'Int32Tensor':
        return int32factory.tensor(im2col(self.value, h_filter, w_filter, padding, strides))

    def conv2d(self, other: Any, strides: int, padding: str='SAME') -> 'Int32Tensor':
        x, y = _lift(self, other)
        return conv2d(x, y, strides, padding)  # type: ignore

    def mod(self, k: int) -> 'Int32Tensor':
        return int32factory.tensor(self.value % k)

    def transpose(self, perm: Union[List[int], Tuple[int]]) -> 'Int32Tensor':
        return int32factory.tensor(tf.transpose(self.value, perm))

    def strided_slice(self, args: Any, kwargs: Any) -> 'Int32Tensor':
        return int32factory.tensor(tf.strided_slice(self.value, *args, **kwargs))

    def split(self, num_split: int, axis: int=0) -> List['Int32Tensor']:
        values = tf.split(self.value, num_split, axis=axis)
        return [int32factory.tensor(value) for value in values]

    def reshape(self, axes: Union[tf.Tensor, List[int]]) -> 'Int32Tensor':
        return int32factory.tensor(tf.reshape(self.value, axes))

    def reduce_sum(self, axis, keepdims) -> 'Int32Tensor':
        return int32factory.tensor(tf.reduce_sum(self.value, axis, keepdims))

    def cumsum(self, axis, exclusive, reverse) -> 'Int32Tensor':
        return int32factory.tensor(tf.cumsum(self.value, axis=axis, exclusive=exclusive, reverse=reverse))

    def equal_zero(self, factory: AbstractFactory=int32factory) -> 'AbstractTensor':
        return factory.tensor(tf.cast(tf.equal(self.value, 0), dtype=factory.native_type))

    def equal(self, other, factory: AbstractFactory=int32factory) -> 'AbstractTensor':
        x, y = _lift(self, other)
        return factory.tensor(tf.cast(tf.equal(x.value, y.value), dtype=factory.native_type))

    def right_shift(self, bitlength):
        return int32factory.tensor(tf.bitwise.right_shift(self.value, bitlength))


class Int32Constant(Int32Tensor, AbstractConstant):

    def __init__(self, value: Union[tf.Tensor, np.ndarray]) -> None:
        v = tf.constant(value, dtype=tf.int32)
        super(Int32Constant, self).__init__(v)

    def __repr__(self) -> str:
        return 'Int32Constant(shape={})'.format(self.shape)


class Int32Placeholder(Int32Tensor, AbstractPlaceholder):

    def __init__(self, shape: List[int]) -> None:
        self.placeholder = tf.placeholder(tf.int32, shape=shape)
        super(Int32Placeholder, self).__init__(self.placeholder)

    def __repr__(self) -> str:
        return 'Int32Placeholder(shape={})'.format(self.shape)

    def feed_from_native(self, value: np.ndarray) -> Dict[tf.Tensor, np.ndarray]:
        assert type(value) in [np.ndarray], type(value)
        return self.feed_from_same(int32factory.tensor(value))

    def feed_from_same(self, value: Int32Tensor) -> Dict[tf.Tensor, np.ndarray]:
        assert isinstance(value, Int32Tensor), type(value)
        return {
            self.placeholder: value.value
        }


class Int32Variable(Int32Tensor, AbstractVariable):

    def __init__(self, initial_value: Union[tf.Tensor, np.ndarray]) -> None:
        self.variable = tf.Variable(initial_value, dtype=tf.int32, trainable=False)
        self.initializer = self.variable.initializer
        super(Int32Variable, self).__init__(self.variable.read_value())

    def __repr__(self) -> str:
        return 'Int32Variable(shape={})'.format(self.shape)

    def assign_from_native(self, value: np.ndarray) -> tf.Operation:
        assert type(value) in [np.ndarray], type(value)
        return self.assign_from_same(int32factory.tensor(value))

    def assign_from_same(self, value: Int32Tensor) -> tf.Operation:
        assert isinstance(value, Int32Tensor), type(value)
        return tf.assign(self.variable, value.value).op
