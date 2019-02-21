from __future__ import absolute_import
from typing import Tuple, Optional
from functools import partial

import abc
import math

import numpy as np
import tensorflow as tf

from ...tensor.factory import AbstractTensor
from ...tensor.shared import binarize
from ...operations import secure_random


def odd_factory(NATIVE_TYPE):

    assert NATIVE_TYPE in (tf.int32, tf.int64)

    class Factory:
        """
        Represents a native integer data type with one value removed in order to obtain an odd modulus.
        More concretely, this data type wraps either tf.int32 or tf.int64 but removes -1 (mapped to 0).
        It is currently not considered for general use but only to support subprotocols of SecureNN.
        """

        def tensor(self, value):
            """
            Wrap `value` in this data type, performing type conversion as needed.
            Internal use should consider explicit construction as an optimization that
            avoids redundant correction.
            """

            if isinstance(value, tf.Tensor):
                if value.dtype is not NATIVE_TYPE:
                    value = tf.cast(value, dtype=NATIVE_TYPE)
                # no assumptions are made about the tensor here and hence we need to
                # apply our mapping for invalid values
                value = _map_minusone_to_zero(value, NATIVE_TYPE)
                return OddDenseTensor(value)

            raise TypeError("Don't know how to handle {}".format(type(value)))

        def constant(self, value):
            raise NotImplementedError()

        def variable(self, initial_value):
            raise NotImplementedError()

        def placeholder(self, shape):
            raise NotImplementedError()

        @property
        def modulus(self):

            if NATIVE_TYPE is tf.int32:
                return 2**32 - 1

            if NATIVE_TYPE is tf.int64:
                return 2**64 - 1

            raise NotImplementedError("Don't know how to handle {}".format(NATIVE_TYPE))

        @property
        def native_type(self):
            return NATIVE_TYPE

        def sample_uniform(self,
                           shape,
                           minval: Optional[int] = None,
                           maxval: Optional[int] = None):
            assert minval is None
            assert maxval is None

            if secure_random.supports_seeded_randomness():
                seed = secure_random.seed()
                return OddUniformTensor(shape=shape, seed=seed)

            elif secure_random.supports_secure_randomness():
                value = _construct_value_from_sampler(sampler=secure_random.random_uniform,
                                                      shape=shape)
                return OddDenseTensor(value)

            else:
                value = _construct_value_from_sampler(sampler=tf.random_uniform,
                                                      shape=shape)
                return OddDenseTensor(value)

        def sample_bounded(self, shape, bitlength: int):
            raise NotImplementedError()

        def stack(self, xs: list, axis: int = 0):
            raise NotImplementedError()

        def concat(self, xs: list, axis: int):
            raise NotImplementedError()

    FACTORY = Factory()

    class OddTensor(AbstractTensor):
        """
        Base class for the concrete odd tensors types. Implements basic functionality needed by
        SecureNN subprotocols from a few abstract properties implemented by concrete types below.
        """

        @property
        def factory(self):
            return FACTORY

        @property
        @abc.abstractproperty
        def value(self) -> tf.Tensor:
            pass

        @property
        @abc.abstractproperty
        def shape(self):
            pass

        def __repr__(self) -> str:
            return '{}(shape={}, native_type={})'.format(
                type(self),
                self.shape,
                NATIVE_TYPE,
            )

        def __getitem__(self, slice):
            return OddDenseTensor(self.value[slice])

        def __add__(self, other):
            return self.add(other)

        def __sub__(self, other):
            return self.sub(other)

        def add(self, other):
            x, y = _lift(self, other)
            bitlength = math.ceil(math.log2(FACTORY.modulus))

            with tf.name_scope('add'):

                # the below avoids redundant seed expansion; can be removed once
                # we have a (per-device) caching mechanism in place
                x_value = x.value
                y_value = y.value

                z = x_value + y_value

                with tf.name_scope('correct_wrap'):

                    # we want to compute whether we wrapped around, ie `pos(x) + pos(y) >= m - 1`,
                    # for correction purposes which, since `m - 1 == 1` for signed integers, can be
                    # rewritten as:
                    #  -> `pos(x) >= m - 1 - pos(y)`
                    #  -> `m - 1 - pos(y) - 1 < pos(x)`
                    #  -> `-1 - pos(y) - 1 < pos(x)`
                    #  -> `-2 - pos(y) < pos(x)`
                    wrapped_around = _lessthan_as_unsigned(-2 - y_value, x_value, bitlength)
                    z += wrapped_around

            return OddDenseTensor(z)

        def sub(self, other):
            x, y = _lift(self, other)
            bitlength = math.ceil(math.log2(FACTORY.modulus))

            with tf.name_scope('sub'):

                # the below avoids redundant seed expansion; can be removed once
                # we have a (per-device) caching mechanism in place
                x_value = x.value
                y_value = y.value

                z = x_value - y_value

                with tf.name_scope('correct-wrap'):

                    # we want to compute whether we wrapped around, ie `pos(x) - pos(y) < 0`,
                    # for correction purposes which can be rewritten as
                    #  -> `pos(x) < pos(y)`
                    wrapped_around = _lessthan_as_unsigned(x_value, y_value, bitlength)
                    z -= wrapped_around

            return OddDenseTensor(z)

        def bits(self, factory=None):
            if factory is None:
                return OddDenseTensor(binarize(self.value))
            else:
                return factory.tensor(binarize(self.value))

        def cast(self, factory):
            if factory is FACTORY:
                return self
            else:
                return factory.tensor(self.value)

    class OddDenseTensor(OddTensor):
        """
        Represents a tensor with explicit values (as opposed to OddUniformTensor with implicit
        values). Internal use only and assume that invalid values have already been mapped.
        """

        def __init__(self, value):
            assert isinstance(value, tf.Tensor)
            self._value = value

        @property
        def value(self) -> tf.Tensor:
            return self._value

        @property
        def shape(self):
            return self._value.shape

    class OddUniformTensor(OddTensor):
        """
        Represents a tensor with uniform values defined implicitly through a seed.
        Internal use only.
        """

        def __init__(self, shape, seed):
            self._seed = seed
            self._shape = shape

        @property
        def shape(self):
            return self._shape

        @property
        def value(self) -> tf.Tensor:
            # TODO(Morten) result should be stored in a (per-device) cache
            with tf.name_scope('expand-seed'):
                value = _construct_value_from_sampler(
                    sampler=partial(secure_random.seeded_random_uniform, seed=self._seed),
                    shape=self._shape)
                return value

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

    def _construct_value_from_sampler(sampler, shape):
        # to get uniform distribution over [min, max] without -1 we sample
        # [min+1, max] and shift negative values down by one
        unshifted_value = sampler(shape=shape,
                                  dtype=NATIVE_TYPE,
                                  minval=NATIVE_TYPE.min + 1,
                                  maxval=NATIVE_TYPE.max)
        value = tf.where(unshifted_value < 0,
                         unshifted_value + tf.ones(shape=unshifted_value.shape,
                                                   dtype=unshifted_value.dtype),
                         unshifted_value)
        return value

    def _lessthan_as_unsigned(x, y, bitlength):
        """
        Performs comparison `x < y` on signed integers *as if* they were unsigned, e.g. `1 < -1`.
        Taken from Section 2-12, page 23, of [Hacker's Delight](https://www.hackersdelight.org/).
        """
        with tf.name_scope('unsigned-compare'):
            not_x = tf.bitwise.invert(x)
            lhs = tf.bitwise.bitwise_and(not_x, y)
            rhs = tf.bitwise.bitwise_and(tf.bitwise.bitwise_or(not_x, y), x - y)
            z = tf.bitwise.right_shift(tf.bitwise.bitwise_or(lhs, rhs), bitlength - 1)
            # turn 0/-1 into 0/1 before returning
            return tf.bitwise.bitwise_and(z, tf.ones(shape=z.shape, dtype=z.dtype))

    def _map_minusone_to_zero(value, native_type):
        """ Maps all -1 values to zero. """
        return tf.where(value == -1, tf.zeros(shape=value.shape, dtype=native_type), value)

    return FACTORY


oddInt32factory = odd_factory(tf.int32)
oddInt64factory = odd_factory(tf.int64)
