from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from typing import Union, Optional, List, Dict, Any, Tuple, Type
from .tensor import AbstractTensor, AbstractVariable, AbstractConstant, AbstractPlaceholder
from .factory import AbstractFactory

from ..config import run


class Tensor(AbstractTensor):

    modulus = 2**31
    int_type = tf.int32

    def __init__(self, value: Union[np.ndarray, tf.Tensor]) -> None:
        self.value = value

    @classmethod
    def from_native(cls, value: Union[np.ndarray, tf.Tensor]) -> 'Tensor':
        assert isinstance(value, (np.ndarray, tf.Tensor)), type(value)
        return cls(value)

    @classmethod
    def from_same(cls, value: 'Tensor') -> 'Tensor':
        assert isinstance(value, Tensor), type(value)
        return cls(value.value)

    def eval(self, sess: tf.Session, feed_dict: Dict[Any, Any]={}, tag: Optional[str]=None) -> 'Tensor':
        concrete_value = run(sess, self.value, feed_dict=feed_dict, tag=tag)
        return Tensor.from_native(concrete_value)

    def to_int32(self) -> Union[tf.Tensor, np.ndarray]:
        return self.value

    @staticmethod
    def sample_uniform(shape: List[int]) -> 'Tensor':
        # TODO[Morten] what should maxval be (account for negative numbers)?
        return Tensor(tf.random_uniform(shape=shape, dtype=tf.int32, maxval=2**31 - 1))

    def __repr__(self) -> str:
        return 'int32.Tensor(shape={})'.format(self.shape)

    @property
    def shape(self) -> Union[Tuple[int, ...], tf.TensorShape]:
        if self.value is None:
            raise Exception("Can't call 'shape' on a empty tensor")

        return self.value.shape

    def __add__(self, other: 'Tensor') -> 'Tensor':
        return self.add(other)

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        return self.sub(other)

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        return self.mul(other)

    def __mod__(self, k: int) -> 'Tensor':
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

    def transpose(self, perm: Union[List[int], Tuple[int]]) -> 'Tensor':
        return Tensor(tf.transpose(self.value, perm))

    def strided_slice(self, args: Any, kwargs: Any) -> 'Tensor':
        return Tensor(tf.strided_slice(self.value, *args, **kwargs))

    def reshape(self, axes: Union[tf.Tensor, List[int]]) -> 'Tensor':
        return Tensor(tf.reshape(self.value, axes))

    @staticmethod
    def stack(x: List['Tensor'], axis: int = 0) -> 'Tensor':
        assert all([isinstance(i, Tensor) for i in x])

        backing = [v.value for v in x]

        return Tensor.from_native(tf.stack(backing, axis=axis))

    @staticmethod
    def concat(x: List['Tensor'], axis: int) -> 'Tensor':
        assert all([isinstance(i, Tensor) for i in x])

        backing = [v.value for v in x]

        return Tensor.from_native(tf.concat(backing, axis=axis))


def _lift(x: Union[Tensor, int]) -> Tensor:
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


class Constant(Tensor, AbstractConstant):

    def __init__(self, value: Union[tf.Tensor, np.ndarray]) -> None:
        super(Constant, self).__init__(tf.constant(value, dtype=tf.int32))

    def __repr__(self) -> str:
        return 'int32.Constant(shape={})'.format(self.shape)


class Placeholder(Tensor, AbstractPlaceholder):

    def __init__(self, shape: List[int]) -> None:
        placeholder = tf.placeholder(tf.int32, shape=shape)
        super(Placeholder, self).__init__(placeholder)
        self.placeholder = placeholder

    def feed_from_native(self, value: np.ndarray) -> Dict[tf.Tensor, np.ndarray]:
        assert type(value) in [np.ndarray], type(value)
        return {
            self.placeholder: value
        }

    def __repr__(self) -> str:
        return 'int32.Placeholder(shape={})'.format(self.shape)


class Variable(Tensor, AbstractVariable):

    def __init__(self, initial_value: Union[tf.Tensor, np.ndarray]) -> None:
        variable = tf.Variable(initial_value, dtype=tf.int32, trainable=False)
        value = variable.read_value()

        super(Variable, self).__init__(value)
        self.variable = variable
        self.initializer = variable.initializer

    def __repr__(self) -> str:
        return 'int32.Variable(shape={})'.format(self.shape)

    def assign_from_native(self, value: np.ndarray) -> tf.Operation:
        assert type(value) in [np.ndarray], type(value)
        return tf.assign(self.variable, value).op

    def assign_from_same(self, value: Tensor) -> tf.Operation:
        assert isinstance(value, Tensor), type(value)
        return tf.assign(self.variable, value.value).op


# type managling to make the below factory work
Int32Tensor = Tensor
Int32Constant = Constant
Int32Variable = Variable
Int32Placeholder = Placeholder


class Int32Factory(AbstractFactory):
    @property
    def Tensor(self) -> Type[Int32Tensor]:
        return Tensor

    @property
    def Constant(self) -> Type[Int32Constant]:
        return Constant

    @property
    def Variable(self) -> Type[Int32Variable]:
        return Variable

    def Placeholder(self, shape: List[int]) -> Int32Placeholder:
        return Placeholder(shape)

    @property
    def modulus(self) -> int:
        return Tensor.modulus
