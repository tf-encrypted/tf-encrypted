from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from typing import Union, Optional, List, Dict, Any

from ..config import run


class Tensor(object):

    modulus = 2**31
    int_type = tf.int32

    def __init__(self, value:Union[np.ndarray,tf.Tensor]) -> None:
        self.value = value

    @staticmethod
    def from_native(value: Union[np.ndarray,tf.Tensor]) -> 'Tensor':
        assert isinstance(value, (np.ndarray,tf.Tensor)), type(value)
        return Tensor(value)
    
    @staticmethod
    def from_same(value: Tensor) -> 'Tensor':
        assert isinstance(value, Tensor), type(value)
        return Tensor(value)

    def eval(self, sess: tf.Session, feed_dict: Dict[Any,Any]={}, tag: Optional[str]=None) -> 'Tensor':
        return Tensor(run(sess, self.value, feed_dict=feed_dict, tag=tag))

    def to_int32(self) -> Union[tf.Tensor,np.ndarray]:
        return self.value

    @staticmethod
    def sample_uniform(shape: List[int]) -> 'Tensor':
        return Tensor(tf.random_uniform(shape=shape, dtype=tf.int32, maxval=2**31-1)) # TODO[Morten] what should maxval be (account for negative numbers)?

    def __repr__(self) -> str:
        return 'int32.Tensor(shape={})'.format(self.shape)

    @property
    def shape(self) -> List[int]:
        if self.value is None:
            raise Exception("Can't call 'shape' on a empty tensor")

        return self.value.shape

    def __add__(self, other: 'Tensor') -> 'Tensor':
        return self.add(other)

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        return self.sub(other)

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        return self.mul(other)

    def __mod__(self, k:int) -> 'Tensor':
        return self.mod(k)

    def add(self, other: 'Tensor') -> 'Tensor':
        x, y = _lift(self), _lift(other)
        return Tensor(x.value + y.value)

    def sub(self, other: 'Tensor') -> 'Tensor':
        x, y = _lift(self), _lift(other)
        return Tensor(x.value - y.value)

    def mul(self, other: 'Tensor') -> 'Tensor':
        x, y = _lift(self), _lift(other)
        return Tensor(x.value * y.value)

    def dot(self, other: 'Tensor') -> 'Tensor':
        x, y = _lift(self), _lift(other)
        return Tensor(tf.matmul(x.value, y.value))

    # def im2col(self, h_filter, w_filter, padding, strides) -> 'Tensor':
    #     return _im2col(self, h_filter, w_filter, padding, strides)

    # def conv2d(self, other, strides, padding='SAME') -> 'Tensor':
    #     return _conv2d(self, other, strides, padding)

    def mod(self, k: int) -> 'Tensor':
        return Tensor(self.value % k)

    def transpose(self, *axes) -> 'Tensor':
        return Tensor(tf.transpose(self.value, axes))

    def strided_slice(self, args: Any, kwargs: Any):
        return Tensor(tf.strided_slice(self.value, *args, **kwargs))

    def reshape(self, *axes) -> 'Tensor':
        return Tensor(tf.reshape(self.value, axes))


def _lift(x):
    # TODO[Morten] support other types of `x`

    if isinstance(x, Tensor):
        return x

    if type(x) is int:
        return Tensor.from_native(np.array([x]))

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

# TODO
# def stack(x: List[Int100Tensor], axis: int = 0):
#     assert all([isinstance(i, Int100Tensor) for i in x])

#     backing = []
#     for i in range(len(x[0].backing)):
#         stacked = [j.backing[i] for j in x]

#         backing.append(tf.stack(stacked, axis=axis))

#     return Int100Tensor.from_decomposed(backing)


class Constant(Tensor):

    @staticmethod
    def from_native(value: np.ndarray) -> 'Constant':
        assert type(value) in [np.ndarray], type(value)
        return Constant(value)

    @staticmethod
    def from_same(value: Tensor) -> 'Constant':
        assert type(value) in [Tensor], type(value)
        return Constant(value.value)

    def __repr__(self) -> str:
        return 'int32.Constant(shape={})'.format(self.shape)


class Placeholder(Tensor):

    def __init__(self, shape):
        placeholder = tf.placeholder(tf.int32, shape=shape)
        super(Placeholder, self).__init__(placeholder)
        self.placeholder = placeholder

    def feed_from_native(self, value):
        assert type(value) in [np.ndarray], type(value)
        return {
            self.placeholder: value
        }

    def __repr__(self):
        return 'int32.Placeholder(shape={})'.format(self.shape)


class Variable(Tensor):

    def __init__(self, initial_value):
        variable = tf.Variable(initial_value, dtype=tf.int32, trainable=False)
        value = variable.read_value()

        super(Variable, self).__init__(value)
        self.variable = variable
        self.initializer = variable.initializer

    @staticmethod
    def from_native(initial_value):
        assert type(initial_value) in [np.ndarray, tf.Tensor], type(initial_value)
        return Variable(initial_value)

    @staticmethod
    def from_same(initial_value):
        assert type(initial_value) in [Tensor], type(initial_value)
        return Variable(initial_value.value)

    def __repr__(self):
        return 'int32.Variable(shape={})'.format(self.shape)

    def assign_from_native(self, value: np.ndarray):
        assert type(value) in [np.ndarray], type(value)
        return tf.assign(self.variable, value).op

    def assign_from_same(self, value: Tensor):
        assert isinstance(value, Tensor), type(value)
        return tf.assign(self.variable, value.value).op
