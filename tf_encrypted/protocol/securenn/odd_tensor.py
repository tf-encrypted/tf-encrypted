from __future__ import absolute_import
from typing import Union, List, Any, Tuple, Type, Optional
import abc
import math

import numpy as np
import tensorflow as tf

from ...tensor.factory import AbstractFactory, AbstractTensor, AbstractConstant
from ...tensor.shared import binarize
from ...operations.secure_random import seeded_random_uniform, seed


SUPPORTED_NATIVE_TYPES = Union[Type['tf.int32'], Type['tf.int64']]


class OddFactory:

    def __init__(self, native_type: SUPPORTED_NATIVE_TYPES) -> None:
        self.native_type = native_type

    @property
    def modulus(self):

        if self.native_type is tf.int32:
            return 2**32 - 1

        if self.native_type is tf.int64:
            return 2**64 - 1

        raise NotImplementedError("Don't know how to handle {}".format(self.native_type))

    def tensor(self, value) -> 'OddTensor':

        if isinstance(value, tf.Tensor):
            # NOTE we make no assumptions about the value here and hence need to apply correction
            if value.dtype is not self.native_type:
                value = tf.cast(value, dtype=self.native_type)
            value = _map_invalid(value, self.native_type)
            return OddDenseTensor(value, self)

        if isinstance(value, OddTensor):
            # NOTE we assumpe that the value here has already been corrected

            assert value.factory == self
            return OddDenseTensor(value.value, self)

        raise TypeError("Don't know how to handle {}".format(type(value)))

    def constant(self, value) -> 'OddTensor':
        raise NotImplementedError()

    def variable(self, initial_value) -> 'OddTensor':
        raise NotImplementedError()

    def placeholder(self, shape) -> 'OddTensor':
        raise NotImplementedError()

    def sample_uniform(self,
                       shape,
                       minval: Optional[int] = None,
                       maxval: Optional[int] = None) -> 'OddTensor':
        assert minval is None
        assert maxval is None
        return OddUniformTensor(shape=shape,
                               seed=seed(),
                               factory=self)

    def sample_bounded(self, shape: List[int], bitlength: int) -> 'OddTensor':
        raise NotImplementedError()

    def stack(self, xs: List['OddTensor'], axis: int = 0) -> 'OddTensor':
        raise NotImplementedError()

    def concat(self, xs: List['OddTensor'], axis: int) -> 'OddTensor':
        raise NotImplementedError()


oddInt32factory = OddFactory(tf.int32)
oddInt64factory = OddFactory(tf.int64)


class OddTensor(AbstractTensor):

    @property
    @abc.abstractproperty
    def value(self) -> tf.Tensor:
        pass

    @property
    @abc.abstractproperty
    def factory(self):
        pass

    @property
    def shape(self):
        return self.value.shape

    def __repr__(self) -> str:
        return '{}(shape={}, native_type={})'.format(
            type(self),
            self.shape,
            self.factory.native_type
        )

    def __getitem__(self, slice: Any) -> Union[tf.Tensor, np.ndarray]:
        return self.factory.tensor(self.value[slice])

    def __add__(self, other) -> 'OddTensor':
        return self.add(other)

    def __sub__(self, other) -> 'OddTensor':
        return self.sub(other)

    def add(self, other) -> 'OddTensor':
        x, y = _lift(self, other)
        bitlength = math.ceil(math.log2(self.factory.modulus))

        with tf.name_scope('add'):
            z = x.value + y.value

            with tf.name_scope('correct-wraparound'):
                # we want to compute whether we wrapped around, ie `pos(x) + pos(y) >= m - 1`,
                # for correction purposes which, since `m - 1 == 1` for signed integers, can
                # be rewritten as
                #  -> `pos(x) >= m - 1 - pos(y)`
                #  -> `m - 1 - pos(y) - 1 < pos(x)`
                #  -> `-1 - pos(y) - 1 < pos(x)`
                #  -> `-2 - pos(y) < pos(x)`
                wrapped_around = _lessthan_as_unsigned(-2 - y.value, x.value, bitlength)
                z += wrapped_around

        return OddDenseTensor(z, self.factory)

    def sub(self, other) -> 'OddTensor':
        x, y = _lift(self, other)
        bitlength = math.ceil(math.log2(self.factory.modulus))

        with tf.name_scope('sub'):
            z = x.value - y.value

            with tf.name_scope('correct-wraparound'):
                # we want to compute whther we wrapped around, ie `pos(x) - pos(y) < 0`,
                # for correction purposes which can be rewritten as
                #  -> `pos(x) < pos(y)`
                wrapped_around = _lessthan_as_unsigned(x.value, y.value, bitlength)
                z -= wrapped_around

        return OddDenseTensor(z, self.factory)

    def bits(self, factory=None) -> AbstractTensor:
        factory = factory or self.factory
        return factory.tensor(binarize(self.value))

    def cast(self, dtype):
        return dtype.tensor(self.value)


class OddDenseTensor(OddTensor):

    def __init__(self, value, factory) -> None:
        """
        This constructor is for internal use where corrections have already been applied.
        """
        self._factory = factory
        self._value = value

    @property
    def value(self) -> tf.Tensor:
        return self._value

    @property
    def shape(self):
        return self._value.shape

    @property
    def factory(self) -> AbstractFactory:
        return self._factory


class OddUniformTensor(OddTensor):

    def __init__(self, shape, seed, factory):
        self._seed = seed
        self._shape = shape
        self._factory = factory

    @property
    def shape(self):
        return self._shape

    @property
    def factory(self) -> AbstractFactory:
        return self._factory

    @property
    def value(self) -> tf.Tensor:
        with tf.name_scope('expand-seed'):
            value = seeded_random_uniform(shape=self._shape,
                                        dtype=self.factory.native_type,
                                        minval=self.factory.native_type.min,
                                        maxval=self.factory.native_type.max,
                                        seed=self._seed)
        # TODO[Morten] mapping introducing bias; fix once secure randomness is in place
        value = _map_invalid(value, self.factory.native_type)
        return value

    def expand(self):
        return OddDenseTensor(self.value, self.factory)


#
# helper methods
#


def _lift(x, y) -> Tuple[OddTensor, OddTensor]:

    if isinstance(x, OddTensor) and isinstance(y, OddTensor):
        assert x.factory == y.factory, "Incompatible types: {} and {}".format(x.factory, y.factory)
        return x, y

    if isinstance(x, OddTensor):

        if isinstance(y, int):
            return x, x.factory.tensor(np.array([y]))

        # if isinstance(y, np.ndarray):
        #     return x, x.factory.tensor(y)

    if isinstance(y, OddTensor):

        if isinstance(x, int):
            return y.factory.tensor(np.array([x])), y

        # if isinstance(x, np.ndarray):
        #     return y.factory.tensor(x), y

    raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))


def _lessthan_as_unsigned(x, y, bitlength):
    # Performs comparison `x < y` on signed integers *as if* they were unsigned, e.g. `1 < -1`.
    # Taken from Section 2-12, page 23, of [Hacker's Delight](https://www.hackersdelight.org/).
    not_x = tf.bitwise.invert(x)
    lhs = tf.bitwise.bitwise_and(not_x, y)
    rhs = tf.bitwise.bitwise_and(tf.bitwise.bitwise_or(not_x, y), x - y)
    z = tf.bitwise.right_shift(tf.bitwise.bitwise_or(lhs, rhs), bitlength - 1)
    # turn 0/-1 into 0/1 before returning
    return tf.bitwise.bitwise_and(z, tf.ones(shape=z.shape, dtype=z.dtype))


def _map_invalid(value, native_type):
    # map all -1 values to zero since the former value is out of bounds in this ring
    with tf.name_scope('map-invalid'):
        return tf.where(value == -1, tf.zeros(shape=value.shape, dtype=native_type), value)
