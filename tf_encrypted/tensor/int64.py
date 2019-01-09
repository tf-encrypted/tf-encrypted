from __future__ import absolute_import
from typing import Union, List, Dict, Tuple, Optional

import numpy as np
import tensorflow as tf

from .factory import (AbstractFactory, AbstractTensor, AbstractVariable,
                      AbstractConstant, AbstractPlaceholder)
from .helpers import inverse
from .shared import binarize, conv2d, im2col
from ..operations.secure_random import seeded_random_uniform, random_uniform

def native_tensor():

    class Factory(AbstractFactory):

        def tensor(self, value):

            if isinstance(value, tf.Tensor):
                if value.dtype is not self.native_type:
                    value = tf.cast(value, dtype=self.native_type)
                return Int64DenseTensor(value)

            if isinstance(value, np.ndarray):
                value = tf.convert_to_tensor(value, dtype=self.native_type)
                return Int64DenseTensor(value)

            raise TypeError("Don't know how to handle {}".format(type(value)))

        def seeded_tensor(self, shape, seed):
            return Int64SeededTensor(shape, seed)

        def constant(self, value):

            if isinstance(value, np.ndarray):
                value = tf.constant(value, dtype=self.native_type)
                return Int64Constant(value)

            raise TypeError("Don't know how to handle {}".format(type(value)))

        def variable(self, initial_value):

            if isinstance(initial_value, (tf.Tensor, np.ndarray)):
                return Int64Variable(initial_value)

            if isinstance(initial_value, Int64DenseTensor):
                return Int64Variable(initial_value.value)

            raise TypeError("Don't know how to handle {}".format(type(initial_value)))

        def placeholder(self, shape):
            return Int64Placeholder(shape)

        @property
        def modulus(self) -> int:
            return 2**64

        @property
        def native_type(self):
            return tf.int64

        def sample_uniform(self,
                           shape,
                           minval: Optional[int] = None,
                           maxval: Optional[int] = None):
            minval = minval or self.native_type.min
            maxval = maxval or self.native_type.max
            value = random_uniform(shape=shape,
                                   dtype=self.native_type,
                                   minval=minval,
                                   maxval=maxval)
            return Int64DenseTensor(value)

        def sample_bounded(self, shape, bitlength: int):
            # TODO[Morten] verify that uses of this work for signed integers
            value = random_uniform(shape=shape,
                                   dtype=self.native_type,
                                   minval=0,
                                   maxval=2**bitlength)
            return Int64DenseTensor(value)

        def stack(self, xs: list, axis: int = 0):
            assert all(isinstance(x, Int64DenseTensor) for x in xs)
            value = tf.stack([x.value for x in xs], axis=axis)
            return Int64DenseTensor(value)

        def concat(self, xs: list, axis: int):
            assert all(isinstance(x, Int64DenseTensor) for x in xs)
            value = tf.concat([x.value for x in xs], axis=axis)
            return Int64DenseTensor(value)


    int64factory = Factory()


    def _lift(x, y) -> Tuple['Int64DenseTensor', 'Int64DenseTensor']:

        if isinstance(x, Int64DenseTensor) and isinstance(y, Int64DenseTensor):
            return x, y

        if isinstance(x, Int64DenseTensor) and isinstance(y, int):
            return x, x.factory.tensor(np.array([y]))

        if isinstance(x, Int64DenseTensor) and isinstance(y, Int64SeededTensor):
            return x, y.expand()

        if isinstance(x, int) and isinstance(y, Int64DenseTensor):
            return y.factory.tensor(np.array([x])), y

        raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))


    class Int64DenseTensor(AbstractTensor):

        def __init__(self, value: tf.Tensor) -> None:
            assert isinstance(value, tf.Tensor)
            self.value = value

        def to_native(self) -> tf.Tensor:
            return self.value

        def bits(self, factory=None) -> AbstractTensor:
            factory = factory or int64factory
            return factory.tensor(binarize(self.value))

        def __repr__(self) -> str:
            return 'Int64DenseTensor(shape={})'.format(self.shape)

        @property
        def shape(self):
            return self.value.shape

        @property
        def factory(self):
            return int64factory

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

        def __neg__(self):
            return self.mul(-1)

        def __getitem__(self, slice):
            return Int64DenseTensor(self.value[slice])

        def add(self, other):
            x, y = _lift(self, other)
            return Int64DenseTensor(x.value + y.value)

        def sub(self, other):
            x, y = _lift(self, other)
            return Int64DenseTensor(x.value - y.value)

        def mul(self, other):
            x, y = _lift(self, other)
            return Int64DenseTensor(x.value * y.value)

        def matmul(self, other):
            x, y = _lift(self, other)
            return Int64DenseTensor(tf.matmul(x.value, y.value))

        def im2col(self, h_filter: int, w_filter: int, padding: str, strides: int):
            return Int64DenseTensor(im2col(self.value, h_filter, w_filter, padding, strides))

        def conv2d(self, other, strides: int, padding: str = 'SAME'):
            x, y = _lift(self, other)
            return conv2d(x, y, strides, padding)  # type: ignore

        def batch_to_space_nd(self, block_shape, crops):
            value = tf.batch_to_space_nd(self.value, block_shape, crops)
            return Int64DenseTensor(value)

        def space_to_batch_nd(self, block_shape, paddings):
            value = tf.space_to_batch_nd(self.value, block_shape, paddings)
            return Int64DenseTensor(value)

        def mod(self, k: int):
            return Int64DenseTensor(self.value % k)

        def transpose(self, perm):
            return Int64DenseTensor(tf.transpose(self.value, perm))

        def strided_slice(self, args, kwargs):
            return Int64DenseTensor(tf.strided_slice(self.value, *args, **kwargs))

        def split(self, num_split: int, axis: int = 0):
            values = tf.split(self.value, num_split, axis=axis)
            return [Int64DenseTensor(value) for value in values]

        def reshape(self, axes: Union[tf.Tensor, List[int]]):
            return Int64DenseTensor(tf.reshape(self.value, axes))

        def reduce_sum(self, axis, keepdims=None):
            return Int64DenseTensor(tf.reduce_sum(self.value, axis, keepdims))

        def cumsum(self, axis, exclusive, reverse):
            return Int64DenseTensor(tf.cumsum(self.value, axis=axis, exclusive=exclusive, reverse=reverse))

        def equal_zero(self, factory=None):
            factory = factory or int64factory
            return factory.tensor(tf.cast(tf.equal(self.value, 0), dtype=factory.native_type))

        def equal(self, other, factory=None):
            x, y = _lift(self, other)
            factory = factory or int64factory
            return factory.tensor(tf.cast(tf.equal(x.value, y.value), dtype=factory.native_type))

        def truncate(self, amount, base=2):
            if base == 2:
                return self.right_shift(amount)
            else:
                factor = base**amount
                factor_inverse = inverse(factor, self.factory.modulus)
                return (self - (self % factor)) * factor_inverse

        def right_shift(self, bitlength):
            return Int64DenseTensor(tf.bitwise.right_shift(self.value, bitlength))

        def expand_dims(self, axis: Optional[int] = None):
            return Int64DenseTensor(tf.expand_dims(self.value, axis))

        def squeeze(self, axis: Optional[List[int]] = None):
            return Int64DenseTensor(tf.squeeze(self.value, axis=axis))

        def negative(self):
            return Int64DenseTensor(tf.negative(self.value))

        def cast(self, factory):
            return factory.tensor(self.value)


    class Int64SeededTensor():
        @property
        def native_type(self):
            return tf.int64

        def __init__(self, shape, seed):
            self.seed = seed
            self.shape = shape

        def expand(self):
            backing = seeded_random_uniform(shape=self.shape,
                                            dtype=self.native_type,
                                            minval=self.native_type.min,
                                            maxval=self.native_type.max,
                                            seed=self.seed)
            return Int64DenseTensor(backing)


    class Int64Constant(Int64DenseTensor, AbstractConstant):

        def __init__(self, constant: tf.Tensor) -> None:
            assert isinstance(constant, tf.Tensor)
            super(Int64Constant, self).__init__(constant)

        def __repr__(self) -> str:
            return 'int64.Constant(shape={})'.format(self.shape)


    class Int64Placeholder(Int64DenseTensor, AbstractPlaceholder):

        def __init__(self, shape: List[int]) -> None:
            self.placeholder = tf.placeholder(tf.int64, shape=shape)
            super(Int64Placeholder, self).__init__(self.placeholder)

        def __repr__(self) -> str:
            return 'Int64Placeholder(shape={})'.format(self.shape)

        def feed(self, value: np.ndarray) -> Dict[tf.Tensor, np.ndarray]:
            assert type(value) in [np.ndarray], type(value)
            return {
                self.placeholder: value
            }


    class Int64Variable(Int64DenseTensor, AbstractVariable):

        def __init__(self, initial_value: Union[tf.Tensor, np.ndarray]) -> None:
            self.variable = tf.Variable(initial_value, dtype=tf.int64, trainable=False)
            self.initializer = self.variable.initializer
            super(Int64Variable, self).__init__(self.variable.read_value())

        def __repr__(self) -> str:
            return 'Int64Variable(shape={})'.format(self.shape)

        def assign_from_native(self, value: np.ndarray) -> tf.Operation:
            assert type(value) in [np.ndarray], type(value)
            return self.assign_from_same(int64factory.tensor(value))

        def assign_from_same(self, value: Int64DenseTensor) -> tf.Operation:
            assert isinstance(value, Int64DenseTensor), type(value)
            return tf.assign(self.variable, value.value).op


    return int64factory


int64factory = native_tensor()
