from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from typing import Union, Optional, List, Dict, Any, Tuple, Type

from ..config import run
from .factory import AbstractFactory
from .tensor import AbstractTensor, AbstractConstant, AbstractVariable

INT_TYPE = tf.int32
bits = 31
p = 67


def native_factory(modulus: int) -> Any:
    class TensorWrap(AbstractTensor):
        @staticmethod
        def from_native(x: Union[tf.Tensor, np.ndarray]) -> 'NativeTensor':
            return NativeTensor.from_native(x, modulus)

        @staticmethod
        def sample_uniform(shape: Union[Tuple[int, ...], tf.TensorShape]) -> 'NativeTensor':
            return NativeTensor(tf.random_uniform(shape=shape, dtype=INT_TYPE, maxval=modulus), modulus)

    class ConstantWrap(TensorWrap, AbstractConstant):
        @staticmethod
        def from_native(x: Union[tf.Tensor, np.ndarray]) -> 'NativeTensor':
            return NativeConstant.from_native(x, modulus)

        @staticmethod
        def from_same(initial_value: 'NativeTensor') -> 'NativeConstant':
            assert type(initial_value) in [NativeTensor], type(initial_value)
            return NativeConstant(initial_value.value, modulus)

    class VariableWrap(TensorWrap, AbstractVariable):
        @staticmethod
        def from_native(x: Union[tf.Tensor, np.ndarray]) -> 'NativeTensor':
            return NativeVariable.from_native(x, modulus)

    class Factory(AbstractFactory):
        @property
        def Tensor(self) -> Type[TensorWrap]:
            return TensorWrap

        @property
        def Constant(self) -> Type[ConstantWrap]:
            return ConstantWrap

        @property
        def Variable(self) -> Type[VariableWrap]:
            return VariableWrap

        def Placeholder(self, shape: List[int]) -> 'NativePlaceholder':  # type: ignore
            return NativePlaceholder(shape, modulus)

        @property
        def modulus(self) -> int:
            return modulus

    return Factory()


class NativeTensor(AbstractTensor):
    int_type = INT_TYPE

    def __init__(self, value: Union[np.ndarray, tf.Tensor], modulus: int) -> None:
        self.value = value
        self.modulus = modulus

    @staticmethod
    def from_native(value: Union[np.ndarray, tf.Tensor], modulus: int) -> 'NativeTensor':
        assert isinstance(value, (np.ndarray, tf.Tensor)), type(value)
        return NativeTensor(value, modulus)

    @staticmethod
    def sample_uniform(shape: Union[Tuple[int, ...], tf.TensorShape], modulus: int) -> 'NativeTensor':
        return NativeTensor(tf.random_uniform(shape=shape, dtype=INT_TYPE, maxval=modulus), modulus)

    @staticmethod
    def sample_bounded(shape: List[int], bitlength: int) -> 'NativeTensor':
        raise NotImplementedError()

    @staticmethod
    def stack(x: List['NativeTensor'], axis: int = 0) -> 'NativeTensor':
        assert all([isinstance(i, NativeTensor) for i in x])

        backing = [v.value for v in x]

        return NativeTensor.from_native(tf.stack(backing, axis=axis), x[0].modulus)

    @staticmethod
    def concat(x: List['NativeTensor'], axis: int) -> 'NativeTensor':
        assert all([isinstance(i, NativeTensor) for i in x])

        backing = [v.value for v in x]

        return NativeTensor.from_native(tf.concat(backing, axis=axis), x[0].modulus)

    def eval(self, sess: tf.Session, feed_dict: Dict[Any, Any]={},
             tag: Optional[str]=None) -> 'NativeTensor':
        return NativeTensor(run(sess, self.value, feed_dict=feed_dict, tag=tag), self.modulus)

    def __getitem__(self, slice: Any) -> Union[tf.Tensor, np.ndarray]:
        return self.value[slice]

    def __repr__(self) -> str:
        return 'NativeTensor({})'.format(self.shape)

    @property
    def shape(self) -> Union[Tuple[int, ...], tf.TensorShape]:
        if self.value is None:
            raise Exception("Can't call 'shape' on a empty tensor")

        return self.value.shape

    def __add__(self, other: Union['NativeTensor', int]) -> 'NativeTensor':
        return self.add(other)

    def __sub__(self, other: Union['NativeTensor', int]) -> 'NativeTensor':
        return self.sub(other)

    def __mul__(self, other: Union['NativeTensor', int]) -> 'NativeTensor':
        return self.mul(other)

    def __mod__(self, k: int) -> 'NativeTensor':
        return self.mod(k)

    def add(self, other: Union['NativeTensor', int]) -> 'NativeTensor':
        x, y = _lift(self, self.modulus), _lift(other, self.modulus)
        return NativeTensor((x.value + y.value) % self.modulus, self.modulus)

    def sub(self, other: Union['NativeTensor', int]) -> 'NativeTensor':
        x, y = _lift(self, self.modulus), _lift(other, self.modulus)
        return NativeTensor((x.value - y.value) % self.modulus, self.modulus)

    def mul(self, other: Union['NativeTensor', int]) -> 'NativeTensor':
        x, y = _lift(self, self.modulus), _lift(other, self.modulus)
        return NativeTensor(x.value * y.value % self.modulus, self.modulus)

    def dot(self, other: Union['NativeTensor', int]) -> 'NativeTensor':
        x, y = _lift(self, self.modulus), _lift(other, self.modulus)
        return NativeTensor(tf.matmul(x.value, y.value) % self.modulus, self.modulus)

    # def im2col(self, h_filter, w_filter, padding, strides) -> 'NativeTensor':
    #     return _im2col(self, h_filter, w_filter, padding, strides)

    # def conv2d(self, other, strides, padding='SAME') -> 'NativeTensor':
    #     return _conv2d(self, other, strides, padding)

    def mod(self, k: int) -> 'NativeTensor':
        x = _lift(self, self.modulus)
        return NativeTensor(x.value % k, self.modulus)

    def transpose(self, perm: Union[List[int], Tuple[int]]) -> 'NativeTensor':
        return NativeTensor(tf.transpose(self.value, perm), self.modulus)

    def strided_slice(self, args: Any, kwargs: Any) -> 'NativeTensor':
        return NativeTensor(tf.strided_slice(self.value, *args, **kwargs), self.modulus)

    def reshape(self, axes: Union[tf.Tensor, List[int]]) -> 'NativeTensor':
        return NativeTensor(tf.reshape(self.value, axes), self.modulus)

    def binarize(self) -> 'NativeTensor':
        bitwidths = tf.range(bits, dtype=INT_TYPE)

        final_shape = [1] * len(self.shape)
        final_shape.append(bits)

        bitwidths = tf.reshape(bitwidths, final_shape)

        val = tf.expand_dims(self.value, -1)
        val = tf.bitwise.bitwise_and(tf.bitwise.right_shift(val, bitwidths), 1)

        return NativeTensor.from_native(val, p)


def _lift(x: Union['NativeTensor', int], modulus: int) -> 'NativeTensor':
    if isinstance(x, NativeTensor):
        return x

    if type(x) is int:
        return NativeTensor.from_native(np.array([x]), modulus)

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


class NativeConstant(NativeTensor):

    def __init__(self, value: Union[tf.Tensor, np.ndarray], modulus: int) -> None:
        v = tf.constant(value, dtype=INT_TYPE)
        super(NativeConstant, self).__init__(v, modulus)

    @staticmethod
    def from_native(value: Union[np.ndarray, tf.Tensor], modulus: int) -> 'NativeConstant':
        assert type(value) in [np.ndarray, tf.Tensor], type(value)
        return NativeConstant(value, modulus)

    @staticmethod
    def from_same(value: NativeTensor, modulus: int) -> 'NativeConstant':
        assert type(value) in [NativeTensor], type(value)
        return NativeConstant(value.value, modulus)

    def __repr__(self) -> str:
        return 'NativeConstant({})'.format(self.shape)


class NativePlaceholder(NativeTensor):

    def __init__(self, shape: List[int], modulus: int) -> None:
        placeholder = tf.placeholder(INT_TYPE, shape=shape)
        super(NativePlaceholder, self).__init__(placeholder, modulus)
        self.placeholder = placeholder

    def feed_from_native(self, value: np.ndarray) -> Dict[tf.Tensor, np.ndarray]:
        assert type(value) in [np.ndarray], type(value)
        return {
            self.placeholder: value
        }

    def feed_from_backing(self, value: 'NativeTensor') -> Dict[tf.Tensor, np.ndarray]:
        assert type(value) in [NativeTensor], type(value)
        assert isinstance(value.value, np.ndarray)

        return {
            self.placeholder: value.value
        }

    def __repr__(self) -> str:
        return 'NativePlaceholder({})'.format(self.shape)


class NativeVariable(NativeTensor):
    def __init__(self, initial_value: Union[tf.Tensor, np.ndarray], modulus: int) -> None:
        variable = tf.Variable(initial_value, dtype=INT_TYPE, trainable=False)
        value: Union[tf.Tensor, np.ndarray] = variable.read_value()

        super(NativeVariable, self).__init__(value, modulus)
        self.variable = variable
        self.initializer = variable.initializer

    @staticmethod
    def from_native(initial_value: Union[np.ndarray, tf.Tensor], modulus: int) -> 'NativeVariable':
        assert type(initial_value) in [np.ndarray, tf.Tensor], type(initial_value)
        return NativeVariable(initial_value, modulus)

    @staticmethod
    def from_same(initial_value: 'NativeTensor', modulus: int) -> 'NativeVariable':
        assert type(initial_value) in [NativeTensor], type(initial_value)
        return NativeVariable(initial_value.value, modulus)

    def __repr__(self) -> str:
        return 'NativeVariable({})'.format(self.shape)

    def assign_from_native(self, value: np.ndarray) -> tf.Operation:
        assert type(value) in [np.ndarray], type(value)
        return tf.assign(self.variable, value).op

    def assign_from_backing(self, value: NativeTensor) -> tf.Operation:
        assert isinstance(value, (NativeTensor,)), type(value)
        return tf.assign(self.variable, value.value).op
