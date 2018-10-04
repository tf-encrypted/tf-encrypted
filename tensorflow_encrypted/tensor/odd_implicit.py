from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from typing import Union, List, Any, Tuple

from .tensor import AbstractTensor


class OddImplicitTensor(AbstractTensor):
    def __init__(self, value: Union[np.ndarray, tf.Tensor], dtype=tf.int32) -> None:
        if dtype not in [tf.int32, tf.int64]:
            raise Exception("Only int32 and int64 dtypes are allowed for the odd implicit tensor")
        self.value = value
        self.dtype = dtype

    @staticmethod
    def from_native(value: Union[np.ndarray, tf.Tensor], dtype=tf.int32) -> 'OddImplicitTensor':
        assert isinstance(value, (np.ndarray, tf.Tensor)), type(value)
        return OddImplicitTensor(value, dtype=dtype)

    @staticmethod
    def sample_uniform(shape: Union[Tuple[int, ...], tf.TensorShape], dtype=tf.int32) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.random_uniform(shape=shape, dtype=dtype,
                                                   minval=dtype.min + 1, maxval=dtype.max), dtype)

    @staticmethod
    def sample_bounded(shape: List[int], bitlength: int) -> 'OddImplicitTensor':
        raise NotImplementedError()

    @staticmethod
    def stack(x: List['OddImplicitTensor'], axis: int = 0) -> 'OddImplicitTensor':
        assert all(isinstance(i, OddImplicitTensor) for i in x)
        return OddImplicitTensor.from_native(tf.stack([v.value for v in x], axis=axis), dtype=x[0].dtype)

    @staticmethod
    def concat(x: List['OddImplicitTensor'], axis: int) -> 'OddImplicitTensor':
        assert all(isinstance(i, OddImplicitTensor) for i in x)
        return OddImplicitTensor.from_native(tf.concat([v.value for v in x], axis=axis), dtype=x[0].dtype)

    def __getitem__(self, slice: Any) -> Union[tf.Tensor, np.ndarray]:
        return self.value[slice]

    def __repr__(self) -> str:
        return 'OddImplicitTensor(shape={}, dtype={})'.format(self.shape, self.dtype)

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
        x, y = self._lift(self), self._lift(other)

        z = x.value + y.value

        z = z + tf.where(
            tf.logical_and(y.value > 0, x.value > tf.int32.max - y.value),
            tf.ones(z.shape, dtype=self.dtype),
            tf.zeros(z.shape, dtype=self.dtype)
        )
        # correct for underflow where needed
        z = z - tf.where(
            tf.logical_and(y.value < 0, x.value < tf.int32.min - y.value),
            tf.ones(z.shape, dtype=self.dtype),
            tf.zeros(z.shape, dtype=self.dtype)
        )

        return OddImplicitTensor(z, dtype=self.dtype)

    def sub(self, other: Union['OddImplicitTensor', int]) -> 'OddImplicitTensor':
        x, y = self._lift(self), self._lift(other)

        z = x.value - y.value

        z = z + tf.where(
            tf.logical_and(y.value < 0, x.value > tf.int32.max + y.value),
            tf.ones(z.shape, dtype=self.dtype),
            tf.zeros(z.shape, dtype=self.dtype)
        )
        # correct for underflow where needed
        z = z - tf.where(
            tf.logical_and(y.value > 0, x.value < tf.int32.min + y.value),
            tf.ones(z.shape, dtype=self.dtype),
            tf.zeros(z.shape, dtype=self.dtype)
        )

        return OddImplicitTensor(z, dtype=self.dtype)

    def mul(self, other: Union['OddImplicitTensor', int]) -> 'OddImplicitTensor':
        raise NotImplementedError()

    def dot(self, other: Union['OddImplicitTensor', int]) -> 'OddImplicitTensor':
        raise NotImplementedError()

    def im2col(self, h_filter, w_filter, padding, strides) -> 'OddImplicitTensor':
        raise NotImplementedError()

    def conv2d(self, other, strides, padding='SAME') -> 'OddImplicitTensor':
        raise NotImplementedError()

    def mod(self, k: int) -> 'OddImplicitTensor':
        x = self._lift(self)
        return OddImplicitTensor(x.value % k, self.dtype)

    def transpose(self, perm: Union[List[int], Tuple[int]]) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.transpose(self.value, perm), self.dtype)

    def strided_slice(self, args: Any, kwargs: Any) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.strided_slice(self.value, *args, **kwargs), self.dtype)

    def reshape(self, axes: Union[tf.Tensor, List[int]]) -> 'OddImplicitTensor':
        return OddImplicitTensor(tf.reshape(self.value, axes), self.dtype)

    def _lift(self, x: Union['OddImplicitTensor', int]) -> 'OddImplicitTensor':
        if isinstance(x, OddImplicitTensor):
            return x

        if type(x) is int:
            return OddImplicitTensor.from_native(np.array([x]), dtype=self.dtype)

        raise TypeError("Unsupported type {}".format(type(x)))
