from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from typing import Union, Optional, List, Dict, Any

from ..config import run

INT_TYPE = tf.int32


class Int32Tensor(object):

    modulus = 2**31
    int_type = INT_TYPE

    def __init__(self, value: Optional[Union[np.ndarray, tf.Tensor]]) -> None:
        self.value = value

    @staticmethod
    def from_native(value: Union[np.ndarray, tf.Tensor]) -> 'Int32Tensor':
        assert isinstance(value, (np.ndarray, tf.Tensor)), type(value)
        return Int32Tensor(value)

    def eval(self, sess: tf.Session, feed_dict: Dict[Any, Any]={}, tag: Optional[str]=None) -> 'Int32Tensor':
        return Int32Tensor(run(sess, self.value, feed_dict=feed_dict, tag=tag))

    def to_int32(self) -> Union[tf.Tensor, np.ndarray]:
        return self.value

    @staticmethod
    def sample_uniform(shape: List[int]) -> 'Int32Tensor':
        return Int32Tensor(tf.random_uniform(shape=shape, dtype=INT_TYPE, maxval=2**32))  # TODO[Morten] account for negative numbers?

    def __repr__(self) -> str:
        return 'Int32Tensor({})'.format(self.shape)

    @property
    def shape(self) -> List[int]:
        return self.value.shape

    def __add__(self, other: 'Int32Tensor') -> 'Int32Tensor':
        return self.add(other)

    def __sub__(self, other: 'Int32Tensor') -> 'Int32Tensor':
        return self.sub(other)

    def __mul__(self, other: 'Int32Tensor') -> 'Int32Tensor':
        return self.mul(other)

    def __mod__(self, k) -> 'Int32Tensor':
        return self.mod(k)

    def add(self, other):
        x, y = _lift(self), _lift(other)
        return Int32Tensor(x.value + y.value)

    def sub(self, other):
        x, y = _lift(self), _lift(other)
        return Int32Tensor(x.value - y.value)

    def mul(self, other: 'Int32Tensor') -> 'Int32Tensor':
        x, y = _lift(self), _lift(other)
        return Int32Tensor(x.value * y.value)

    def dot(self, other: 'Int32Tensor') -> 'Int32Tensor':
        x, y = _lift(self), _lift(other)
        return Int32Tensor(tf.matmul(x.value, y.value))

    # def im2col(self, h_filter, w_filter, padding, strides) -> 'Int32Tensor':
    #     return _im2col(self, h_filter, w_filter, padding, strides)

    # def conv2d(self, other, strides, padding='SAME') -> 'Int32Tensor':
    #     return _conv2d(self, other, strides, padding)

    def mod(self, k) -> 'Int32Tensor':
        return Int32Tensor(self.value % k)

    def transpose(self, *axes) -> 'Int32Tensor':
        return Int32Tensor(tf.transpose(self.value, axes))

    def strided_slice(self, args: Any, kwargs: Any):
        return _strided_slice(self, args, kwargs)

    def reshape(self, *axes) -> 'Int32Tensor':
        return _reshape(self, *axes)


def _lift(x):
    # TODO[Morten] support other types of `x`

    if isinstance(x, Int32Tensor):
        return x

    if type(x) is int:
        return Int32Tensor.from_native(np.array([x]))

    raise TypeError("Unsupported type {}".format(type(x)))


# TODO
# def _im2col(x, h_filter, w_filter, padding, strides):
#     assert isinstance(x, Int100Tensor), type(x)
#     backing = _crt_im2col(x.backing, h_filter, w_filter, padding, strides)
#     return Int100Tensor.from_decomposed(backing)

# TODO
# def _conv2d(x, y, strides, padding):
#     assert isinstance(x, Int100Tensor), type(x)
#     assert isinstance(y, Int100Tensor), type(y)

#     h_filter, w_filter, d_filters, n_filters = map(int, y.shape)
#     n_x, d_x, h_x, w_x = map(int, x.shape)
#     if padding == "SAME":
#         h_out = int(math.ceil(float(h_x) / float(strides)))
#         w_out = int(math.ceil(float(w_x) / float(strides)))
#     if padding == "VALID":
#         h_out = int(math.ceil(float(h_x - h_filter + 1) / float(strides)))
#         w_out = int(math.ceil(float(w_x - w_filter + 1) / float(strides)))

#     X_col = x.im2col(h_filter, w_filter, padding, strides)
#     W_col = y.transpose(3, 2, 0, 1).reshape(int(n_filters), -1)
#     out = W_col.dot(X_col)

#     out = out.reshape(n_filters, h_out, w_out, n_x)
#     out = out.transpose(3, 0, 1, 2)

#     return out


def _strided_slice(x: Int32Tensor, args: Any, kwargs: Any):
    return Int32Tensor(tf.strided_slice(x.value, *args, **kwargs))


def _reshape(x: Int32Tensor, *axes):
    return Int32Tensor(tf.reshape(x.value, axes))


# TODO
# def stack(x: List[Int100Tensor], axis: int = 0):
#     assert all([isinstance(i, Int100Tensor) for i in x])

#     backing = []
#     for i in range(len(x[0].backing)):
#         stacked = [j.backing[i] for j in x]

#         backing.append(tf.stack(stacked, axis=axis))

#     return Int100Tensor.from_decomposed(backing)


class Int32Constant(Int32Tensor):

    def __init__(self, value: np.ndarray) -> None:
        value = tf.constant(value, dtype=INT_TYPE)
        super(Int32Constant, self).__init__(value)

    @staticmethod
    def from_native(value: np.ndarray) -> 'Int32Constant':
        assert type(value) in [np.ndarray, tf.Tensor], type(value)
        return Int32Constant(value)

    def __repr__(self) -> str:
        return 'Int32Constant({})'.format(self.shape)


class Int32Placeholder(Int32Tensor):

    def __init__(self, shape):
        placeholder = tf.placeholder(INT_TYPE, shape=shape)
        super(Int32Placeholder, self).__init__(placeholder)
        self.placeholder = placeholder

    def feed_from_native(self, value):
        assert type(value) in [np.ndarray], type(value)
        return {
            self.placeholder: value
        }

    def __repr__(self):
        return 'Int32Placeholder({})'.format(self.shape)


class Int32Variable(Int32Tensor):

    def __init__(self, initial_value):
        variable = tf.Variable(initial_value, dtype=INT_TYPE, trainable=False)
        value = variable.read_value()

        super(Int32Variable, self).__init__(value)
        self.variable = variable
        self.initializer = variable.initializer

    @staticmethod
    def from_native(initial_value):
        assert type(initial_value) in [np.ndarray, tf.Tensor], type(initial_value)
        return Int32Variable(initial_value)

    def __repr__(self):
        return 'Int32Variable({})'.format(self.shape)

    def assign_from_native(self, value: np.ndarray):
        assert type(value) in [np.ndarray], type(value)
        return tf.assign(self.variable, value).op

    def assign_from_same(self, value: Int32Tensor):
        assert isinstance(value, (Int32Tensor,)), type(value)
        return tf.assign(self.variable, value.value).op
