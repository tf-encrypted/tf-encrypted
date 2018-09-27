from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from typing import Union, Optional, List, Dict, Any, Tuple, Type

from ..config import run
from .factory import AbstractFactory
from .tensor import AbstractTensor, AbstractConstant, AbstractVariable

INT_TYPE = tf.int32


class PrimeTensor(AbstractTensor):

    int_type = INT_TYPE

    def __init__(self, value: Union[np.ndarray, tf.Tensor], modulus: int) -> None:
        self.value = value
        self.modulus = modulus

    @staticmethod
    def from_native(value: Union[np.ndarray, tf.Tensor], modulus: int) -> 'PrimeTensor':
        assert isinstance(value, (np.ndarray, tf.Tensor)), type(value)
        return PrimeTensor(value, modulus)

    @staticmethod
    def sample_uniform(shape: Union[Tuple[int, ...], tf.TensorShape], modulus: int) -> 'PrimeTensor':
        return PrimeTensor(tf.random_uniform(shape=shape, dtype=INT_TYPE, minval=0, maxval=modulus), modulus)

    @staticmethod
    def sample_bounded(shape: List[int], bitlength: int) -> 'PrimeTensor':
        raise NotImplementedError()

    @staticmethod
    def stack(x: List['PrimeTensor'], axis: int = 0) -> 'PrimeTensor':
        assert all(isinstance(i, PrimeTensor) for i in x)
        return PrimeTensor.from_native(tf.stack([v.value for v in x], axis=axis), x[0].modulus)

    @staticmethod
    def concat(x: List['PrimeTensor'], axis: int) -> 'PrimeTensor':
        assert all(isinstance(i, PrimeTensor) for i in x)
        return PrimeTensor.from_native(tf.concat([v.value for v in x], axis=axis), x[0].modulus)

    def eval(self, sess: tf.Session, feed_dict: Dict[Any, Any]={},
             tag: Optional[str]=None) -> 'PrimeTensor':
        return PrimeTensor(run(sess, self.value, feed_dict=feed_dict, tag=tag), self.modulus)

    def __getitem__(self, slice: Any) -> Union[tf.Tensor, np.ndarray]:
        return PrimeTensor.from_native(self.value[slice], self.modulus)

    def __repr__(self) -> str:
        return 'PrimeTensor(shape={}, modulus={})'.format(self.shape, self.modulus)

    @property
    def shape(self) -> Union[Tuple[int, ...], tf.TensorShape]:
        return self.value.shape

    def __add__(self, other: Union['PrimeTensor', int]) -> 'PrimeTensor':
        return self.add(other)

    def __sub__(self, other: Union['PrimeTensor', int]) -> 'PrimeTensor':
        return self.sub(other)

    def __mul__(self, other: Union['PrimeTensor', int]) -> 'PrimeTensor':
        return self.mul(other)

    def __mod__(self, k: int) -> 'PrimeTensor':
        return self.mod(k)

    def add(self, other: Union['PrimeTensor', int]) -> 'PrimeTensor':
        x, y = _lift(self, self.modulus), _lift(other, self.modulus)
        return PrimeTensor((x.value + y.value) % self.modulus, self.modulus)

    def sub(self, other: Union['PrimeTensor', int]) -> 'PrimeTensor':
        x, y = _lift(self, self.modulus), _lift(other, self.modulus)
        return PrimeTensor((x.value - y.value) % self.modulus, self.modulus)

    def mul(self, other: Union['PrimeTensor', int]) -> 'PrimeTensor':
        x, y = _lift(self, self.modulus), _lift(other, self.modulus)
        return PrimeTensor(x.value * y.value % self.modulus, self.modulus)

    def dot(self, other: Union['PrimeTensor', int]) -> 'PrimeTensor':
        x, y = _lift(self, self.modulus), _lift(other, self.modulus)
        return PrimeTensor(tf.matmul(x.value, y.value) % self.modulus, self.modulus)

    def im2col(self, h_filter, w_filter, padding, strides) -> 'PrimeTensor':
        raise NotImplementedError()

    def conv2d(self, other, strides, padding='SAME') -> 'PrimeTensor':
        raise NotImplementedError()

    def mod(self, k: int) -> 'PrimeTensor':
        x = _lift(self, self.modulus)
        return PrimeTensor(x.value % k, self.modulus)

    def transpose(self, perm: Union[List[int], Tuple[int]]) -> 'PrimeTensor':
        return PrimeTensor(tf.transpose(self.value, perm), self.modulus)

    def strided_slice(self, args: Any, kwargs: Any) -> 'PrimeTensor':
        return PrimeTensor(tf.strided_slice(self.value, *args, **kwargs), self.modulus)

    def reshape(self, axes: Union[tf.Tensor, List[int]]) -> 'PrimeTensor':
        return PrimeTensor(tf.reshape(self.value, axes), self.modulus)

    def compute_wrap(self, y: AbstractTensor, modulus: int) -> AbstractTensor:
        return PrimeTensor(tf.cast(self.value + y.value >= modulus, dtype=tf.int32), self.modulus)


def _lift(x: Union['PrimeTensor', int], modulus: int) -> 'PrimeTensor':
    if isinstance(x, PrimeTensor):
        return x

    if type(x) is int:
        return PrimeTensor.from_native(np.array([x]), modulus)

    raise TypeError("Unsupported type {}".format(type(x)))


class PrimeConstant(PrimeTensor):

    def __init__(self, value: Union[tf.Tensor, np.ndarray], modulus: int) -> None:
        v = tf.constant(value, dtype=INT_TYPE)
        super(PrimeConstant, self).__init__(v, modulus)

    @staticmethod
    def from_native(value: Union[np.ndarray, tf.Tensor], modulus: int) -> 'PrimeConstant':
        assert type(value) in [np.ndarray, tf.Tensor], type(value)
        return PrimeConstant(value, modulus)

    @staticmethod
    def from_same(value: PrimeTensor, modulus: int) -> 'PrimeConstant':
        assert type(value) in [PrimeTensor], type(value)
        return PrimeConstant(value.value, modulus)

    def __repr__(self) -> str:
        return 'PrimeConstant({})'.format(self.shape)


class PrimePlaceholder(PrimeTensor):

    def __init__(self, shape: List[int], modulus: int) -> None:
        placeholder = tf.placeholder(INT_TYPE, shape=shape)
        super(PrimePlaceholder, self).__init__(placeholder, modulus)
        self.placeholder = placeholder

    def feed_from_native(self, value: np.ndarray) -> Dict[tf.Tensor, np.ndarray]:
        assert type(value) in [np.ndarray], type(value)
        return {
            self.placeholder: value
        }

    def feed_from_backing(self, value: 'PrimeTensor') -> Dict[tf.Tensor, np.ndarray]:
        assert type(value) in [PrimeTensor], type(value)
        assert isinstance(value.value, np.ndarray)
        return {
            self.placeholder: value.value
        }

    def __repr__(self) -> str:
        return 'PrimePlaceholder({})'.format(self.shape)


class PrimeVariable(PrimeTensor, AbstractVariable):

    def __init__(self, initial_value: Union[tf.Tensor, np.ndarray], modulus: int) -> None:
        variable = tf.Variable(initial_value, dtype=INT_TYPE, trainable=False)
        value = variable.read_value()

        super(PrimeVariable, self).__init__(value, modulus)
        self.variable = variable
        self.initializer = variable.initializer

    @staticmethod
    def from_native(initial_value: Union[np.ndarray, tf.Tensor], modulus: int) -> 'PrimeVariable':
        assert type(initial_value) in [np.ndarray, tf.Tensor], type(initial_value)
        return PrimeVariable(initial_value, modulus)

    @staticmethod
    def from_same(initial_value: 'PrimeTensor', modulus: int) -> 'PrimeVariable':
        assert type(initial_value) in [PrimeTensor], type(initial_value)
        return PrimeVariable(initial_value.value, modulus)

    def __repr__(self) -> str:
        return 'PrimeVariable({})'.format(self.shape)

    def assign_from_native(self, value: np.ndarray) -> tf.Operation:
        assert type(value) in [np.ndarray], type(value)
        return tf.assign(self.variable, value).op

    def assign_from_backing(self, value: PrimeTensor) -> tf.Operation:
        assert isinstance(value, (PrimeTensor,)), type(value)
        return tf.assign(self.variable, value.value).op


def prime_factory(modulus: int) -> Any:

    class TensorWrap(AbstractTensor):
        @staticmethod
        def from_native(x: Union[tf.Tensor, np.ndarray]) -> PrimeTensor:
            return PrimeTensor.from_native(x, modulus)

        @staticmethod
        def sample_uniform(shape: Union[Tuple[int, ...], tf.TensorShape]) -> PrimeTensor:
            return PrimeTensor(tf.random_uniform(shape=shape, dtype=INT_TYPE, maxval=modulus), modulus)

    class ConstantWrap(TensorWrap, AbstractConstant):
        @staticmethod
        def from_native(x: Union[tf.Tensor, np.ndarray]) -> PrimeTensor:
            return PrimeConstant.from_native(x, modulus)

        @staticmethod
        def from_same(initial_value: PrimeTensor) -> PrimeConstant:
            return PrimeConstant.from_same(initial_value.value, modulus)

    class VariableWrap(TensorWrap, AbstractVariable):
        @staticmethod
        def from_native(x: Union[tf.Tensor, np.ndarray]) -> PrimeTensor:
            return PrimeVariable.from_native(x, modulus)

        @staticmethod
        def from_same(initial_value: PrimeTensor) -> PrimeVariable:
            return PrimeVariable.from_same(initial_value, modulus)

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

        def Placeholder(self, shape: List[int]) -> PrimePlaceholder:  # type: ignore
            return PrimePlaceholder(shape, modulus)

        @property
        def modulus(self) -> int:
            return modulus

    return Factory()
