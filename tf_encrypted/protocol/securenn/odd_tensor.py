from __future__ import absolute_import
from typing import List, Tuple, Optional

import abc
import math

import numpy as np
import tensorflow as tf

from ...tensor.factory import AbstractTensor
from ...tensor.shared import binarize
from ...operations.secure_random import seeded_random_uniform, seed


class OddTensor(AbstractTensor):
    """
    Base class for the concrete odd tensors types. Implements basic functionality needed by
    SecureNN subprotocols from a few abstract properties implemented by concrete types below.
    """

    @property
    @abc.abstractproperty
    def value(self) -> tf.Tensor:
        pass

    @property
    @abc.abstractproperty
    def factory(self):
        pass

    @property
    @abc.abstractproperty
    def shape(self):
        pass

    def __repr__(self) -> str:
        return '{}(shape={}, native_type={})'.format(
            type(self),
            self.shape,
            self.factory.native_type,
        )

    def __getitem__(self, slice):
        return OddDenseTensor(self.value[slice], self.factory)

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def add(self, other):
        x, y = _lift(self, other)
        bitlength = math.ceil(math.log2(self.factory.modulus))

        # the below avoids redundant seed expansion; can be removed once
        # we have a (per-device) caching mechanism in place
        x_value = x.value
        y_value = y.value

        z = x_value + y_value

        # we want to compute whether we wrapped around, ie `pos(x) + pos(y) >= m - 1`,
        # for correction purposes which, since `m - 1 == 1` for signed integers, can be
        # rewritten as:
        #  -> `pos(x) >= m - 1 - pos(y)`
        #  -> `m - 1 - pos(y) - 1 < pos(x)`
        #  -> `-1 - pos(y) - 1 < pos(x)`
        #  -> `-2 - pos(y) < pos(x)`
        wrapped_around = _lessthan_as_unsigned(-2 - y_value, x_value, bitlength)
        z += wrapped_around

        return OddDenseTensor(z, self.factory)

    def sub(self, other):
        x, y = _lift(self, other)
        bitlength = math.ceil(math.log2(self.factory.modulus))

        # the below avoids redundant seed expansion; can be removed once
        # we have a (per-device) caching mechanism in place
        x_value = x.value
        y_value = y.value

        z = x_value - y_value

        # we want to compute whether we wrapped around, ie `pos(x) - pos(y) < 0`,
        # for correction purposes which can be rewritten as
        #  -> `pos(x) < pos(y)`
        wrapped_around = _lessthan_as_unsigned(x_value, y_value, bitlength)
        z -= wrapped_around

        return OddDenseTensor(z, self.factory)

    def bits(self, factory=None):
        if factory is None:
            return OddDenseTensor(binarize(self.value), self.factory)
        else:
            return factory.tensor(binarize(self.value))

    def cast(self, factory):
        if factory == self.factory:
            return self
        else:
            return factory.tensor(self.value)


class OddDenseTensor(OddTensor):
    """
    Represents a tensor with explicit values (as opposed to OddUniformTensor with implicit
    values). Internal use only and assume that invalid values have already been mapped.
    """

    def __init__(self, value, factory):
        assert isinstance(value, tf.Tensor)
        assert value.dtype == factory.native_type
        self._value = value
        self._factory = factory

    @property
    def factory(self):
        return self._factory

    @property
    def value(self) -> tf.Tensor:
        return self._value

    @property
    def shape(self):
        return self.value.shape


class OddUniformTensor(OddTensor):
    """
    Represents a tensor with uniform values defined implicitly through a seed.
    Internal use only.
    """

    def __init__(self, shape, seed, factory):
        self._seed = seed
        self._shape = shape
        self._factory = factory

    @property
    def factory(self):
        return self._factory

    @property
    def shape(self):
        return self._shape

    @property
    def value(self) -> tf.Tensor:
        # TODO(Morten) result should be stored in a (per-device) cache
        with tf.name_scope('expand-seed'):
            # to get uniform distribution over [min, max] without -1 we sample
            # [min+1, max] and shift negative values down by one
            unshifted_value = seeded_random_uniform(shape=self._shape,
                                                    dtype=self._factory.native_type,
                                                    minval=self._factory.native_type.min + 1,
                                                    maxval=self._factory.native_type.max,
                                                    seed=self._seed)
            value = tf.where(unshifted_value < 0,
                             unshifted_value + tf.ones(shape=unshifted_value.shape,
                                                       dtype=unshifted_value.dtype),
                             unshifted_value)
            return value

    def expand(self):
        return OddDenseTensor(self.value, self.factory)


class Factory:
    """
    Represents a native integer data type with one value removed in order to obtain an odd modulus.
    More concretely, this data type wraps either tf.int32 or tf.int64 but removes -1 (mapped to 0).
    It is currently not considered for general use but only to support subprotocols of SecureNN.
    """

    def __init__(self, native_type):
        assert native_type in (tf.int32, tf.int64)
        self.native_type = native_type

    def tensor(self, value) -> OddTensor:
        """
        Wrap `value` in this data type, performing type conversion as needed.
        Internal use should consider explicit construction as an optimization that
        avoids redundant correction.
        """

        if isinstance(value, tf.Tensor):
            if value.dtype is not self.native_type:
                value = tf.cast(value, dtype=self.native_type)
            # no assumptions are made about the tensor here and hence we need to
            # apply our mapping for invalid values
            value = _map_minusone_to_zero(value, self.native_type)
            return OddDenseTensor(value, self)

        raise TypeError("Don't know how to handle {}".format(type(value)))

    def constant(self, value) -> OddTensor:
        raise NotImplementedError()

    def variable(self, initial_value) -> OddTensor:
        raise NotImplementedError()

    def placeholder(self, shape) -> OddTensor:
        raise NotImplementedError()

    @property
    def modulus(self):

        if self.native_type is tf.int32:
            return 2**32 - 1

        if self.native_type is tf.int64:
            return 2**64 - 1

        raise NotImplementedError("Don't know how to handle {}".format(self.native_type))

    def sample_uniform(self,
                       shape,
                       minval: Optional[int] = None,
                       maxval: Optional[int] = None) -> OddTensor:
        assert minval is None
        assert maxval is None
        return OddUniformTensor(shape=shape,
                                seed=seed(),
                                factory=self)

    def sample_bounded(self, shape, bitlength: int) -> OddTensor:
        raise NotImplementedError()

    def stack(self, xs: List[OddTensor], axis: int = 0) -> OddTensor:
        raise NotImplementedError()

    def concat(self, xs: List[OddTensor], axis: int) -> OddTensor:
        raise NotImplementedError()


oddInt32factory = Factory(tf.int32)
oddInt64factory = Factory(tf.int64)


#
# internal helper methods
#


def _lift(x, y) -> Tuple[OddTensor, OddTensor]:
    """
    Attempts to lift x and y to compatible OddTensors for further processing.
    """

    if isinstance(x, OddTensor) and isinstance(y, OddTensor):
        assert x.factory == y.factory, "Incompatible types: {} and {}".format(x.factory, y.factory)
        return x, y

    if isinstance(x, OddTensor):

        if isinstance(y, int):
            return x, x.factory.tensor(np.array([y]))

    if isinstance(y, OddTensor):

        if isinstance(x, int):
            return y.factory.tensor(np.array([x])), y

    raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))


def _lessthan_as_unsigned(x, y, bitlength):
    """
    Performs comparison `x < y` on signed integers *as if* they were unsigned, e.g. `1 < -1`.
    Taken from Section 2-12, page 23, of [Hacker's Delight](https://www.hackersdelight.org/).
    """
    not_x = tf.bitwise.invert(x)
    lhs = tf.bitwise.bitwise_and(not_x, y)
    rhs = tf.bitwise.bitwise_and(tf.bitwise.bitwise_or(not_x, y), x - y)
    z = tf.bitwise.right_shift(tf.bitwise.bitwise_or(lhs, rhs), bitlength - 1)
    # turn 0/-1 into 0/1 before returning
    return tf.bitwise.bitwise_and(z, tf.ones(shape=z.shape, dtype=z.dtype))


def _map_minusone_to_zero(value, native_type):
    """ Maps all -1 values to zero. """
    return tf.where(value == -1, tf.zeros(shape=value.shape, dtype=native_type), value)
