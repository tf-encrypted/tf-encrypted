from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from typing import Union, Optional, List, Dict, Any, Tuple, Type
from .tensor import AbstractTensor, AbstractVariable, AbstractConstant, AbstractPlaceholder
from .factory import AbstractFactory
from .prime import PrimeTensor
from .native_shared import binarize, conv2d, im2col

from ..config import run


class Int64Tensor(AbstractTensor):

    modulus = 2**64
    int_type = tf.int64

    def __init__(self, value: Union[np.ndarray, tf.Tensor]) -> None:
        self.value = value

    @classmethod
    def from_native(cls, value: Union[np.ndarray, tf.Tensor]) -> 'Int64Tensor':
        assert isinstance(value, (np.ndarray, tf.Tensor)), type(value)
        return cls(value)

    @classmethod
    def from_same(cls, value: 'Int64Tensor') -> 'Int64Tensor':
        assert isinstance(value, Int64Tensor), type(value)
        return cls(value.value)

    def eval(self, sess: tf.Session, feed_dict: Dict[Any, Any]={}, tag: Optional[str]=None) -> 'Int64Tensor':
        concrete_value = run(sess, self.value, feed_dict=feed_dict, tag=tag)
        return Int64Tensor.from_native(concrete_value)

    def to_native(self) -> Union[tf.Tensor, np.ndarray]:
        return self.value

    def to_bits(self) -> PrimeTensor:
        return PrimeTensor.from_native(binarize(self.value), 67)

    @staticmethod
    def sample_uniform(shape: List[int]) -> 'Int64Tensor':
        return Int64Tensor(tf.random_uniform(shape=shape, dtype=tf.int64, minval=tf.int64.min, maxval=tf.int64.max))

    def __repr__(self) -> str:
        return 'int64.Tensor(shape={})'.format(self.shape)

    @property
    def shape(self) -> Union[Tuple[int, ...], tf.TensorShape]:
        return self.value.shape

    def __add__(self, other: Any) -> 'Int64Tensor':
        return self.add(other)

    def __sub__(self, other: Any) -> 'Int64Tensor':
        return self.sub(other)

    def __mul__(self, other: Any) -> 'Int64Tensor':
        return self.mul(other)

    def __mod__(self, k: int) -> 'Int64Tensor':
        return self.mod(k)

    def add(self, other: Any) -> 'Int64Tensor':
        x, y = Int64Tensor.lift(self), Int64Tensor.lift(other)
        return Int64Tensor(x.value + y.value)

    def sub(self, other: Any) -> 'Int64Tensor':
        x, y = Int64Tensor.lift(self), Int64Tensor.lift(other)
        return Int64Tensor(x.value - y.value)

    def mul(self, other: Any) -> 'Int64Tensor':
        x, y = Int64Tensor.lift(self), Int64Tensor.lift(other)
        return Int64Tensor(x.value * y.value)

    def matmul(self, other: Any) -> 'Int64Tensor':
        x, y = Int64Tensor.lift(self), Int64Tensor.lift(other)
        return Int64Tensor(tf.matmul(x.value, y.value))

    def im2col(self, h_filter: int, w_filter: int, padding: str, strides: int) -> 'Int64Tensor':
        return im2col(self, h_filter, w_filter, padding, strides)

    def conv2d(self, other: Any, strides: int, padding: str='SAME') -> 'Int64Tensor':
        x, y = Int64Tensor.lift(self), Int64Tensor.lift(other)
        return conv2d(x, y, strides, padding)  # type: ignore

    def mod(self, k: int) -> 'Int64Tensor':
        return Int64Tensor(self.value % k)

    def transpose(self, perm: Union[List[int], Tuple[int]]) -> 'Int64Tensor':
        return Int64Tensor(tf.transpose(self.value, perm))

    def strided_slice(self, args: Any, kwargs: Any) -> 'Int64Tensor':
        return Int64Tensor(tf.strided_slice(self.value, *args, **kwargs))

    def reshape(self, axes: Union[tf.Tensor, List[int]]) -> 'Int64Tensor':
        return Int64Tensor(tf.reshape(self.value, axes))

    @staticmethod
    def stack(xs: List['Int64Tensor'], axis: int = 0) -> 'Int64Tensor':
        assert all(isinstance(x, Int64Tensor) for x in xs)
        return Int64Tensor(tf.stack([x.value for x in xs], axis=axis))

    @staticmethod
    def concat(xs: List['Int64Tensor'], axis: int) -> 'Int64Tensor':
        assert all(isinstance(x, Int64Tensor) for x in xs)
        return Int64Tensor(tf.concat([x.value for x in xs], axis=axis))


class Int64Constant(Int64Tensor, AbstractConstant):

    def __init__(self, value: Union[tf.Tensor, np.ndarray]) -> None:
        super(Int64Constant, self).__init__(tf.constant(value, dtype=tf.int64))

    def __repr__(self) -> str:
        return 'int64.Constant(shape={})'.format(self.shape)


class Int64Placeholder(Int64Tensor, AbstractPlaceholder):

    def __init__(self, shape: List[int]) -> None:
        placeholder = tf.placeholder(tf.int64, shape=shape)
        super(Int64Placeholder, self).__init__(placeholder)
        self.placeholder = placeholder

    def feed_from_native(self, value: np.ndarray) -> Dict[tf.Tensor, np.ndarray]:
        assert type(value) in [np.ndarray], type(value)
        return {
            self.placeholder: value
        }

    def __repr__(self) -> str:
        return 'int64.Placeholder(shape={})'.format(self.shape)


class Int64Variable(Int64Tensor, AbstractVariable):

    def __init__(self, initial_value: Union[tf.Tensor, np.ndarray]) -> None:
        variable = tf.Variable(initial_value, dtype=tf.int64, trainable=False)
        value = variable.read_value()

        super(Int64Variable, self).__init__(value)
        self.variable = variable
        self.initializer = variable.initializer

    def __repr__(self) -> str:
        return 'int64.Variable(shape={})'.format(self.shape)

    def assign_from_native(self, value: np.ndarray) -> tf.Operation:
        assert type(value) in [np.ndarray], type(value)
        return tf.assign(self.variable, value).op

    def assign_from_same(self, value: Int64Tensor) -> tf.Operation:
        assert isinstance(value, Int64Tensor), type(value)
        return tf.assign(self.variable, value.value).op


class Int64Factory(AbstractFactory):
    @property
    def Tensor(self) -> Type[Int64Tensor]:
        return Int64Tensor

    @property
    def Constant(self) -> Type[Int64Constant]:
        return Int64Constant

    @property
    def Variable(self) -> Type[Int64Variable]:
        return Int64Variable

    def Placeholder(self, shape: List[int]) -> Int64Placeholder:
        return Int64Placeholder(shape)

    @property
    def modulus(self) -> int:
        return Int64Tensor.modulus
