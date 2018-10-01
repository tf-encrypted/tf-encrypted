from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from typing import Union, Optional, List, Dict, Any, Tuple, Type

from ..config import run
from .tensor import AbstractTensor


class OddImplicitTensor(AbstractTensor):
    def __init__(self, value: Union[np.ndarray, tf.Tensor], modulus: int, dtype=tf.int32) -> None:
        if dtype is not tf.int32 and dtype is not tf.int64:
            raise Exception("Only int32 and int64 dtypes are allowed for the odd implicit tensor")
        self.value = value
        self.dtype = tf.int32
        self.modulus = modulus

    @staticmethod
    def from_native(value: Union[np.ndarray, tf.Tensor], modulus: int, dtype=tf.int32) -> 'OddImplicitTensor':
        assert isinstance(value, (np.ndarray, tf.Tensor)), type(value)
        return OddImplicitTensor(value, modulus, dtype=dtype)

    @staticmethod
    def sample_uniform(shape: Union[Tuple[int, ...], tf.TensorShape], modulus: int, dtype=tf.int32) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.random_uniform(shape=shape, dtype=dtype,
                                                   minval=dtype.min, maxval=dtype.max), modulus, dtype)

    @staticmethod
    def sample_bounded(shape: List[int], bitlength: int) -> 'OddImplicitTensor':
        raise NotImplementedError()

    @staticmethod
    def stack(x: List['OddImplicitTensor'], axis: int = 0) -> 'OddImplicitTensor':
        assert all(isinstance(i, OddImplicitTensor) for i in x)
        return OddImplicitTensor.from_native(tf.stack([v.value for v in x], axis=axis), x[0].modulus)

    @staticmethod
    def concat(x: List['OddImplicitTensor'], axis: int) -> 'OddImplicitTensor':
        assert all(isinstance(i, OddImplicitTensor) for i in x)
        return OddImplicitTensor.from_native(tf.concat([v.value for v in x], axis=axis), x[0].modulus)

    def eval(self, sess: tf.Session, feed_dict: Dict[Any, Any]={},
             tag: Optional[str]=None) -> 'OddImplicitTensor':
        return OddImplicitTensor(run(sess, self.value, feed_dict=feed_dict, tag=tag), self.modulus)

    def __getitem__(self, slice: Any) -> Union[tf.Tensor, np.ndarray]:
        return self.value[slice]

    def __repr__(self) -> str:
        return 'OddImplicitTensor(shape={}, modulus={})'.format(self.shape, self.modulus)

    @property
    def shape(self) -> Union[Tuple[int, ...], tf.TensorShape]:
        return self.value.shape

    def __add__(self, other: Union['OddImplicitTensor', int]) -> 'OddImplicitTensor':
        return self.add(other)

    def __sub__(self, other: Union['OddImplicitTensor', int]) -> 'OddImplicitTensor':
        return self.sub(other)

    def __mul__(self, other: Union['OddImplicitTensor', int]) -> 'OddImplicitTensor':
        return self.mul(other)

    def __mod__(self, k: int) -> 'OddImplicitTensor':
        return self.mod(k)

    def add(self, other: Union['OddImplicitTensor', int]) -> 'OddImplicitTensor':
        x, y = _lift(self, self.modulus, self.dtype), _lift(other, self.modulus, self.dtype)

        z = x.value + y.value

        def func(val):
            return tf.cond(val >= self.modulus, lambda: val + 1, lambda: val)

        ret = tf.map_fn(func, z)

        return OddImplicitTensor(ret, self.modulus, dtype=self.dtype)

    def sub(self, other: Union['OddImplicitTensor', int]) -> 'OddImplicitTensor':
        x, y = _lift(self, self.modulus, self.dtype), _lift(other, self.modulus, self.dtype)

        z = x.value - y.value

        def func(val):
            return tf.cond(val < self.modulus, lambda: val, lambda: val + 1)

        ret = tf.map_fn(func, z)

        return OddImplicitTensor(ret, self.modulus, dtype=self.dtype)

    def mul(self, other: Union['OddImplicitTensor', int]) -> 'OddImplicitTensor':
        raise NotImplementedError()

    def dot(self, other: Union['OddImplicitTensor', int]) -> 'OddImplicitTensor':
        raise NotImplementedError()

    def im2col(self, h_filter, w_filter, padding, strides) -> 'OddImplicitTensor':
        raise NotImplementedError()

    def conv2d(self, other, strides, padding='SAME') -> 'OddImplicitTensor':
        raise NotImplementedError()

    def mod(self, k: int) -> 'OddImplicitTensor':
        x = _lift(self, self.modulus)
        return OddImplicitTensor(x.value % k, self.modulus)

    def transpose(self, perm: Union[List[int], Tuple[int]]) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.transpose(self.value, perm), self.modulus)

    def strided_slice(self, args: Any, kwargs: Any) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.strided_slice(self.value, *args, **kwargs), self.modulus)

    def reshape(self, axes: Union[tf.Tensor, List[int]]) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.reshape(self.value, axes), self.modulus)


def _lift(x: Union['OddImplicitTensor', int], modulus: int, dtype) -> 'OddImplicitTensor':
    if isinstance(x, OddImplicitTensor):
        return x

    if type(x) is int:
        return OddImplicitTensor.from_native(np.array([x]), modulus, dtype=dtype)

    raise TypeError("Unsupported type {}".format(type(x)))
