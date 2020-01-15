"""
Use TensorFlow's native bool type.
"""
from __future__ import absolute_import
from typing import Union, List, Dict, Tuple, Optional
import abc

import numpy as np
import tensorflow as tf

from .factory import AbstractFactory
from .factory import AbstractTensor
from .factory import AbstractVariable
from .factory import AbstractConstant
from .factory import AbstractPlaceholder
from ..operations import secure_random as crypto


def bool_factory():
  """Constructs the native tensor Factory."""

  class Factory(AbstractFactory):
    """Native tensor factory."""

    def tensor(self, value):
      if isinstance(value, Tensor):
        return value

      if isinstance(value, tf.Tensor):
        if value.dtype is not self.native_type:
          value = tf.cast(value, dtype=self.native_type)
        return DenseTensor(value)

      value = np.array(value)
      value = tf.convert_to_tensor(value)
      value = tf.cast(value, self.native_type)
      return DenseTensor(value)

    def constant(self, value):
      value = tf.constant(value, dtype=self.native_type)
      return Constant(value)

    def variable(self, initial_value):
      if isinstance(initial_value, (tf.Tensor, np.ndarray)):
        return Variable(initial_value)

      if isinstance(initial_value, Tensor):
        return Variable(initial_value.value)

      msg = "Don't know how to handle {}"
      raise TypeError(msg.format(type(initial_value)))

    def placeholder(self, shape):
      return Placeholder(shape)

    @property
    def native_type(self):
      return tf.bool

    def modulus(self) -> int:
      return 2

    def sample_uniform(self, shape):
      minval = 0
      maxval = 2

      if crypto.supports_seeded_randomness():
        seed = crypto.secure_seed()
        return UniformTensor(shape=shape, seed=seed, minval=minval, maxval=maxval)

      if crypto.supports_secure_randomness():
        sampler = crypto.random_uniform
      else:
        sampler = tf.random_uniform
      value = sampler(shape=shape, minval=minval, maxval=maxval, dtype=tf.int32)
      value = tf.cast(value, tf.bool)
      return DenseTensor(value)

    def sample_seeded_uniform(self, shape, seed):
      minval = 0
      maxval = 2

      if crypto.supports_seeded_randomness():
        # Don't use UniformTensor for lazy sampling here, because the `seed` might be something (e.g., key) we
        # want to protect, and we cannot send it to another party
        value = crypto.seeded_random_uniform(
            shape=shape,
            dtype=tf.int32,
            minval=minval,
            maxval=maxval,
            seed=seed,
        )
        value = tf.cast(value, tf.bool)
        return DenseTensor(value)
      else:
        raise NotImplementedError(
            "Secure seeded randomness implementation is not available.")

    def sample_bounded(self, shape, bitlength: int):
      raise NotImplementedError("No bounded sampling for boolean type.")

    def stack(self, xs: list, axis: int = 0):
      assert all(isinstance(x, Tensor) for x in xs)
      value = tf.stack([x.value for x in xs], axis=axis)
      return DenseTensor(value)

    def concat(self, xs: list, axis: int):
      assert all(isinstance(x, Tensor) for x in xs)
      value = tf.concat([x.value for x in xs], axis=axis)
      return DenseTensor(value)

    def where(self, condition, x, y, v2=True):
      if not isinstance(condition, tf.Tensor):
        msg = "Don't know how to handle `condition` of type {}"
        raise TypeError(msg.format(type(condition)))
      if not v2:
        value = tf.where(condition, x.value, y.value)
      else:
        value = tf.where_v2(condition, x.value, y.value)
      return DenseTensor(value)

  def _lift(x, y) -> Tuple['Tensor', 'Tensor']:

    if isinstance(x, Tensor) and isinstance(y, Tensor):
      return x, y

    if isinstance(x, Tensor):
      return x, x.factory.tensor(y)

    if isinstance(y, Tensor):
      return y.factory.tensor(x), y

    raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))

  class Tensor(AbstractTensor):
    """Base class for other native tensor classes."""

    @property
    @abc.abstractproperty
    def value(self):
      pass

    @property
    @abc.abstractproperty
    def shape(self):
      pass

    def identity(self):
      value = tf.identity(self.value)
      return DenseTensor(value)

    def to_native(self) -> tf.Tensor:
      return self.value

    def __repr__(self) -> str:
      return '{}(shape={})'.format(type(self), self.shape)

    @property
    def factory(self):
      return FACTORY

    @property
    def dtype(self):
      return self.factory.native_type

    def __getitem__(self, slc):
      return DenseTensor(self.value[slc])

    def transpose(self, perm):
      return DenseTensor(tf.transpose(self.value, perm))

        def reshape(self, axes: Union[tf.Tensor, List[int]]):
            return DenseTensor(tf.reshape(self.value, axes))
    def strided_slice(self, args, kwargs):
      return DenseTensor(tf.strided_slice(self.value, *args, **kwargs))

        def equal(self, other, factory=None):
            x, y = _lift(self, other)
            factory = factory or FACTORY
            return factory.tensor(tf.cast(tf.equal(x.value, y.value),
                                          dtype=factory.native_type))
    def gather(self, indices: list, axis: int = 0):
      return DenseTensor(tf.gather(self.value, indices, axis=axis))

        def expand_dims(self, axis: Optional[int] = None):
            return DenseTensor(tf.expand_dims(self.value, axis))
    def split(self, num_split: int, axis: int = 0):
      values = tf.split(self.value, num_split, axis=axis)
      return [DenseTensor(value) for value in values]

        def squeeze(self, axis: Optional[List[int]] = None):
            return DenseTensor(tf.squeeze(self.value, axis=axis))

        def cast(self, factory):
            return factory.tensor(self.value)

        def __xor__(self, other):
            return self.xor(other)

        def xor(self, other):
            x, y = _lift(self, other)
            value = tf.math.logical_xor(x.value, y.value)
            return DenseTensor(value)

        def __and__(self, other):
            return self.and_(other)

        def and_(self, other):
            # Because "and" is a keyword in Python, the naming "and_" follows the way how Python handles this:
            # https://docs.python.org/3.4/library/operator.html
            x, y = _lift(self, other)
            value = tf.math.logical_and(x.value, y.value)
            return DenseTensor(value)

        def __invert__(self):
            return self.invert()

        def invert(self):
            value = tf.math.logical_not(self.value)
            return DenseTensor(value)

    class DenseTensor(Tensor):
        """Public native Tensor class."""

        def __init__(self, value):
            self._value = value

        @property
        def shape(self):
            return self._value.shape

        @property
        def value(self):
            return self._value

        @property
        def support(self):
            return [self._value]

    class UniformTensor(Tensor):
        """Class representing a uniform-random, lazily sampled tensor.

        Lazy sampling optimizes communication by sending seeds in place of
        fully-expanded tensors."""

        def __init__(self, shape, seed, minval, maxval):
            self._seed = seed
            self._shape = shape
            self._minval = minval
            self._maxval = maxval

        @property
        def shape(self):
            return self._shape

        @property
        def value(self):
            with tf.name_scope('expand-seed'):
                return tf.cast(
                    crypto.seeded_random_uniform(
                        shape=self._shape,
                        dtype=tf.int32,
                        minval=self._minval,
                        maxval=self._maxval,
                        seed=self._seed),
                    tf.bool)

        @property
        def support(self):
            return [self._seed]

    class Constant(DenseTensor, AbstractConstant):
        """Native Constant class."""

        def __init__(self, constant: tf.Tensor) -> None:
            assert isinstance(constant, tf.Tensor)
            super(Constant, self).__init__(constant)

        def __repr__(self) -> str:
            return 'Constant(shape={})'.format(self.shape)

    class Placeholder(DenseTensor, AbstractPlaceholder):
        """Native Placeholder class."""

        def __init__(self, shape: List[int]) -> None:
            self.placeholder = tf.placeholder(tf.bool, shape=shape)
            super(Placeholder, self).__init__(self.placeholder)

        def __repr__(self) -> str:
            return 'Placeholder(shape={})'.format(self.shape)

        def feed(self, value: np.ndarray) -> Dict[tf.Tensor, np.ndarray]:
            assert isinstance(value, np.ndarray), type(value)
            return {
                self.placeholder: value
            }

    class Variable(DenseTensor, AbstractVariable):
        """Native Variable class."""

        def __init__(self, initial_value: Union[tf.Tensor, np.ndarray]) -> None:
            self.variable = tf.Variable(
                initial_value, dtype=tf.bool, trainable=False)
            self.initializer = self.variable.initializer
            super(Variable, self).__init__(self.variable.read_value())

        def __repr__(self) -> str:
            return 'Variable(shape={})'.format(self.shape)

        def assign_from_native(self, value: np.ndarray) -> tf.Operation:
            assert isinstance(value, np.ndarray), type(value)
            return self.assign_from_same(FACTORY.tensor(value))

        def assign_from_same(self, value: Tensor) -> tf.Operation:
            assert isinstance(value, Tensor), type(value)
            return tf.assign(self.variable, value.value).op

    FACTORY = Factory()
    return FACTORY
