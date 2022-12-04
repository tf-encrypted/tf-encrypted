"""
Use TensorFlow's native bool type.
"""
from __future__ import absolute_import

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

from ..operations import secure_random as crypto
from .factory import AbstractConstant
from .factory import AbstractFactory
from .factory import AbstractTensor
from .factory import AbstractVariable


def bool_factory():
    """Constructs the native tensor Factory."""

    class Factory(AbstractFactory):
        """Native tensor factory."""

        def tensor(self, initial_value, encode: bool=True):
            if encode:
                initial_value = self._encode(initial_value)
            return Tensor(initial_value)

        def constant(self, initial_value, encode: bool=True):
            if encode:
                initial_value = self._encode(initial_value)
            return Constant(initial_value)

        def variable(self, initial_value, encode: bool=True):
            if isinstance(initial_value, Tensor):
                initial_value = initial_value.value
                encode = False
            if encode:
                initial_value = self._encode(initial_value)
            variable_value = tf.Variable(
                initial_value, dtype=self.native_type, trainable=False
            )
            return Variable(variable_value)
        
        def _encode(self, scaled_value):
            if isinstance(scaled_value, (int, float)):
                scaled_value = np.array(scaled_value)
                return tf.convert_to_tensor(scaled_value, dtype=self.native_type)
            elif isinstance(scaled_value, np.ndarray):
                return tf.convert_to_tensor(scaled_value, dtype=self.native_type)
            elif isinstance(scaled_value, tf.Tensor):
                return tf.cast(scaled_value, dtype=self.native_type)
            else:
                raise TypeError(
                    "Don't know how to handle {}".format(type(scaled_value))
                )
        
        def _decode(self, encode_value):
            if isinstance(encode_value, tf.Tensor):
                return encode_value
            else:
                raise TypeError(
                    "Don't know how to handle {}".format(type(encode_value))
                )

        @property
        def native_type(self):
            return tf.bool

        @property
        def modulus(self) -> int:
            return 2

        @property
        def nbits(self):
            return 0

        def sample_uniform(self, shape):  # pylint: disable=arguments-differ
            minval = 0
            maxval = 2

            if crypto.supports_seeded_randomness():
                value = crypto.seeded_random_uniform(
                    shape=shape,
                    dtype=tf.int32,
                    minval=minval,
                    maxval=maxval,
                    seed=crypto.secure_seed(),
                )
                value = tf.cast(value, tf.bool)
                return Tensor(value)

            if crypto.supports_secure_randomness():
                sampler = crypto.random_uniform
            else:
                sampler = tf.random.uniform
            value = sampler(shape=shape, minval=minval, maxval=maxval, dtype=tf.int32)
            value = tf.cast(value, tf.bool)
            return Tensor(value)

        def sample_seeded_uniform(self, shape, seed):
            """Seeded sample of a random tensor.

            Arguments:
                shape (tuple of ints), shape of the tensor to sample
                seed (int), seed for the sampler to use

            Returns a tensor of shape `shape` drawn from a uniform distribution over
            the interval [0,2].
            """
            minval = 0
            maxval = 2

            if crypto.supports_seeded_randomness():
                # Don't use UniformTensor for lazy sampling here, because the `seed`
                # might be something (e.g., key) we want to protect, and we cannot
                # send it to another party
                value = crypto.seeded_random_uniform(
                    shape=shape,
                    dtype=tf.int32,
                    minval=minval,
                    maxval=maxval,
                    seed=seed,
                )
                value = tf.cast(value, tf.bool)
                return Tensor(value)
            else:
                value = tf.random.stateless_uniform(
                    shape, seed, minval=minval, maxval=maxval, dtype=tf.int32
                )
                value = tf.cast(value, tf.bool)
                return Tensor(value)

        def sample_bounded(self, shape, bitlength: int):
            raise NotImplementedError("No bounded sampling for boolean type.")

        def stack(self, xs: list, axis: int = 0):
            assert all(isinstance(x, Tensor) for x in xs)
            value = tf.stack([x.value for x in xs], axis=axis)
            return Tensor(value)

        def concat(self, xs: list, axis: int):
            assert all(isinstance(x, Tensor) for x in xs)
            value = tf.concat([x.value for x in xs], axis=axis)
            return Tensor(value)

        def where(self, condition, x, y):
            if not isinstance(condition, tf.Tensor):
                msg = "Don't know how to handle `condition` of type {}"
                raise TypeError(msg.format(type(condition)))
            value = tf.where(condition, x.value, y.value)
            return Tensor(value)

    def _lift(x, y) -> Tuple["Tensor", "Tensor"]:  # noqa:F821

        if isinstance(x, Tensor) and isinstance(y, Tensor):
            return x, y

        if isinstance(x, Tensor):
            return x, x.factory.tensor(y)

        if isinstance(y, Tensor):
            return y.factory.tensor(x), y

        raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))

    class Tensor(AbstractTensor):
        """Base class for other native tensor classes."""

        def __init__(self, value: tf.Tensor):
            self._value = value

        @property
        def value(self):
            return self._value

        @property
        def shape(self):
            return self._value.shape

        def identity(self):
            value = tf.identity(self.value)
            return Tensor(value)

        def to_native(self) -> tf.Tensor:
            return self.factory._decode(self.value)

        def __repr__(self) -> str:
            return "{}(shape={})".format(type(self), self.shape)

        @property
        def factory(self):
            return FACTORY

        @property
        def device(self):
            return self._value.device

        @property
        def dtype(self):
            return self.factory.native_type

        def __getitem__(self, slc):
            return Tensor(self.value[slc])

        def transpose(self, perm):
            return Tensor(tf.transpose(self.value, perm))

        def strided_slice(self, args, kwargs):
            return Tensor(tf.strided_slice(self.value, *args, **kwargs))

        def gather(self, indices: list, axis: int = 0):
            return Tensor(tf.gather(self.value, indices, axis=axis))

        def split(self, num_split: int, axis: int = 0):
            values = tf.split(self.value, num_split, axis=axis)
            return [Tensor(value) for value in values]

        def reshape(self, axes: Union[tf.Tensor, List[int]]):
            return Tensor(tf.reshape(self.value, axes))

        def equal(self, other, factory=None):
            x, y = _lift(self, other)
            factory = factory or FACTORY
            return factory.tensor(
                tf.cast(tf.equal(x.value, y.value), dtype=factory.native_type)
            )

        def expand_dims(self, axis: Optional[int] = None):
            return Tensor(tf.expand_dims(self.value, axis))

        def squeeze(self, axis: Optional[List[int]] = None):
            return Tensor(tf.squeeze(self.value, axis=axis))

        def cast(self, factory):
            return factory.tensor(self.value)

        def __xor__(self, other):
            return self.logical_xor(other)

        def logical_xor(self, other):
            x, y = _lift(self, other)
            value = tf.math.logical_xor(x.value, y.value)
            return Tensor(value)

        def __and__(self, other):
            return self.logical_and(other)

        def logical_and(self, other):
            x, y = _lift(self, other)
            value = tf.math.logical_and(x.value, y.value)
            return Tensor(value)

        def __or__(self, other):
            return self.logical_or(other)

        def logical_or(self, other):
            x, y = _lift(self, other)
            value = tf.math.logical_or(x.value, y.value)
            return Tensor(value)

        def __invert__(self):
            return self.logical_not()

        def logical_not(self):
            value = tf.math.logical_not(self.value)
            return Tensor(value)

    class Constant(Tensor, AbstractConstant):
        """Native Constant class."""

        def __init__(self, constant_value: tf.Tensor) -> None:
            super(Constant, self).__init__(constant_value)

        def __repr__(self) -> str:
            return "Constant(shape={})".format(self.shape)

    class Variable(Tensor, AbstractVariable):
        """Native Variable class."""

        def __init__(self, variable_value: tf.Tensor) -> None:
            self.variable = variable_value
            super(Variable, self).__init__(self.variable.read_value())

        def __repr__(self) -> str:
            return "Variable(shape={})".format(self.shape)

        def assign(self, value: Union[Tensor, np.ndarray]) -> None:
            if isinstance(value, Tensor):
                self.variable.assign(value.value)
            if isinstance(value, np.ndarray):
                self.variable.assign(value)

            raise TypeError("Don't know how to handle {}".format(type(value)))

        def read_value(self) -> Tensor:
            return Tensor(self.variable.read_value())

    FACTORY = Factory()  # pylint: disable=invalid-name

    return FACTORY
