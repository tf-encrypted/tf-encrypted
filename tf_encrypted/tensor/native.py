"""Native tensors and their factory.

These use TensorFlow's native dtypes tf.int32 and tf.int64 for the given float
encoding being used (fixed-point, etc.)."""

import math
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

from ..operations import aux
from ..operations import secure_random
from .factory import AbstractConstant
from .factory import AbstractFactory
from .factory import AbstractTensor
from .factory import AbstractVariable
from .helpers import inverse
from .shared import binarize
from .shared import conv2d
from .shared import im2col
from .shared import im2patches
from .shared import patches2im


def native_factory(
    NATIVE_TYPE,
    EXPLICIT_MODULUS=None,
):  # pylint: disable=invalid-name
    """Constructs the native tensor Factory."""

    class Factory(AbstractFactory):
        """Native tensor factory."""

        def tensor(self, initial_value):

            if isinstance(initial_value, Tensor):
                return initial_value
            elif isinstance(initial_value, tf.Tensor):
                if initial_value.dtype is not self.native_type:
                    initial_value = tf.cast(initial_value, dtype=self.native_type)
                    return Tensor(initial_value)
                else:
                    return Tensor(tf.identity(initial_value))
            elif isinstance(initial_value, np.ndarray):
                initial_value = tf.convert_to_tensor(
                    initial_value, dtype=self.native_type
                )
                return Tensor(initial_value)
            else:
                raise TypeError(
                    "Don't know how to handle {}".format(type(initial_value))
                )

        def constant(self, initial_value):
            if isinstance(initial_value, Tensor):
                return initial_value
            elif isinstance(initial_value, tf.Tensor):
                if initial_value.dtype is not self.native_type:
                    initial_value = tf.cast(initial_value, dtype=self.native_type)
                    return Constant(initial_value)
                else:
                    return Constant(tf.identity(initial_value))
            elif isinstance(initial_value, np.ndarray):
                constant_value = tf.constant(initial_value, dtype=self.native_type)
                return Constant(constant_value)
            else:
                raise TypeError(
                    "Don't know how to handle {}".format(type(initial_value))
                )

        def variable(self, initial_value):

            if isinstance(initial_value, tf.Variable):
                return Variable(initial_value)
            elif isinstance(initial_value, (tf.Tensor, np.ndarray)):
                variable_value = tf.Variable(
                    initial_value, dtype=self.native_type, trainable=False
                )
                return Variable(variable_value)
            elif isinstance(initial_value, Tensor):
                variable_value = tf.Variable(
                    initial_value.value, dtype=self.native_type, trainable=False
                )
                return Variable(variable_value)
            else:
                raise TypeError(
                    "Don't know how to handle {}".format(type(initial_value))
                )

        @property
        def min(self):
            if EXPLICIT_MODULUS is not None:
                return 0
            return NATIVE_TYPE.min

        @property
        def max(self):
            if EXPLICIT_MODULUS is not None:
                return EXPLICIT_MODULUS
            return NATIVE_TYPE.max

        @property
        def modulus(self) -> int:
            if EXPLICIT_MODULUS is not None:
                return EXPLICIT_MODULUS
            return NATIVE_TYPE.max - NATIVE_TYPE.min + 1

        @property
        def native_type(self):
            return NATIVE_TYPE

        @property
        def nbits(self):
            return NATIVE_TYPE.size * 8

        def sample_uniform(
            self, shape, minval: Optional[int] = None, maxval: Optional[int] = None
        ):
            minval = self.min if minval is None else minval
            # TODO(Morten) believe this should be native_type.max+1
            maxval = self.max if maxval is None else maxval

            if secure_random.supports_seeded_randomness():
                value = secure_random.seeded_random_uniform(
                    shape=shape,
                    dtype=NATIVE_TYPE,
                    minval=minval,
                    maxval=maxval,
                    seed=secure_random.secure_seed(),
                )
                return Tensor(value)

            if secure_random.supports_secure_randomness():
                sampler = secure_random.random_uniform
            else:
                sampler = tf.random.uniform
            value = sampler(
                shape=shape, minval=minval, maxval=maxval, dtype=NATIVE_TYPE
            )
            return Tensor(value)

        def sample_seeded_uniform(
            self,
            shape,
            seed,
            minval: Optional[int] = None,
            maxval: Optional[int] = None,
        ):
            """Seeded sample of a random tensor.

            Arguments:
                shape (tuple of ints), shape of the tensor to sample
                seed (int), seed for the sampler to use
                minval (int), the a in the interval [a,b]
                maxval (int), the b in the interval [a,b]

            Returns a tensor of shape `shape` drawn from a uniform distribution over
            the interval [minval,maxval].
            """
            minval = self.min if minval is None else minval
            maxval = self.max if maxval is None else maxval

            if secure_random.supports_seeded_randomness():
                # Don't use UniformTensor for lazy sampling here, because the `seed`
                # might be something (e.g., key) we want to protect, and we cannot
                # send it to another party
                value = secure_random.seeded_random_uniform(
                    shape=shape,
                    dtype=NATIVE_TYPE,
                    minval=minval,
                    maxval=maxval,
                    seed=seed,
                )
                return Tensor(value)
            else:
                if NATIVE_TYPE not in (tf.int32, tf.int64):
                    value = tf.random.stateless_uniform(
                        shape, seed[6:8], minval=minval, maxval=maxval, dtype=tf.int32
                    )
                    value = tf.cast(value, NATIVE_TYPE)
                else:
                    value = tf.random.stateless_uniform(
                        shape,
                        seed[6:8],
                        minval=minval,
                        maxval=maxval,
                        dtype=NATIVE_TYPE,
                    )
                return Tensor(value)

        def sample_bounded(self, shape, bitlength: int):
            maxval = 2**bitlength
            assert maxval <= self.max

            if secure_random.supports_seeded_randomness():
                value = secure_random.seeded_random_uniform(
                    shape=shape,
                    dtype=NATIVE_TYPE,
                    minval=0,
                    maxval=maxval,
                    seed=secure_random.secure_seed(),
                )
                return Tensor(value)

            if secure_random.supports_secure_randomness():
                sampler = secure_random.random_uniform
            else:
                sampler = tf.random.uniform
            value = sampler(shape=shape, minval=0, maxval=maxval, dtype=NATIVE_TYPE)
            return Tensor(value)

        def sample_bits(self, shape):
            return self.sample_bounded(shape, bitlength=1)

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

        def tile(self, x, multiples):
            return Tensor(tf.tile(x.value, multiples))

        def scatter_nd(self, indices, updates, shape):
            return Tensor(tf.scatter_nd(indices, updates.value, shape))

    FACTORY = Factory()

    def _lift(x, y) -> Tuple["Tensor", "Tensor"]:  # noqa:F821

        if isinstance(x, Tensor) and isinstance(y, Tensor):
            return x, y

        if isinstance(x, Tensor):

            if isinstance(y, int):
                return x, x.factory.tensor(np.array(y))

        if isinstance(y, Tensor):

            if isinstance(x, int):
                return y.factory.tensor(np.array(x)), y

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
            return self.value

        def bits(self, factory=None, bitsize=None) -> AbstractTensor:
            factory = factory or FACTORY
            if EXPLICIT_MODULUS is None:
                return factory.tensor(binarize(self.value, bitsize))
            if bitsize is None:
                bitsize = math.ceil(math.log2(EXPLICIT_MODULUS))
            return factory.tensor(binarize(self.value % EXPLICIT_MODULUS, bitsize))

        def __repr__(self) -> str:
            return "{}(shape={})".format(type(self), self.shape)

        @property
        def factory(self):
            return FACTORY

        def __add__(self, other):
            x, y = _lift(self, other)
            return x.add(y)

        def __radd__(self, other):
            x, y = _lift(self, other)
            return y.add(x)

        def __sub__(self, other):
            x, y = _lift(self, other)
            return x.sub(y)

        def __rsub__(self, other):
            x, y = _lift(self, other)
            return y.sub(x)

        def __mul__(self, other):
            x, y = _lift(self, other)
            return x.mul(y)

        def __rmul__(self, other):
            x, y = _lift(self, other)
            return x.mul(y)

        def __mod__(self, k: int):
            return self.mod(k)

        def __neg__(self):
            return self.mul(-1)

        def __getitem__(self, slc):
            return Tensor(self.value[slc])

        def add(self, other):
            x, y = _lift(self, other)
            value = x.value + y.value
            if EXPLICIT_MODULUS is not None:
                value %= EXPLICIT_MODULUS
            return Tensor(value)

        def sub(self, other):
            x, y = _lift(self, other)
            value = x.value - y.value
            if EXPLICIT_MODULUS is not None:
                value %= EXPLICIT_MODULUS
            return Tensor(value)

        def mul(self, other):
            x, y = _lift(self, other)
            value = x.value * y.value
            if EXPLICIT_MODULUS is not None:
                value %= EXPLICIT_MODULUS
            return Tensor(value)

        def matmul(self, other):
            x, y = _lift(self, other)
            value = tf.matmul(x.value, y.value)
            if EXPLICIT_MODULUS is not None:
                value %= EXPLICIT_MODULUS
            return Tensor(value)

        def bit_gather(self, start, stride):
            value = aux.bit_gather(self.value, start, stride)
            return Tensor(value)

        def bit_split_and_gather(self, stride):
            value = aux.bit_split_and_gather(self.value, stride)
            return Tensor(value)

        def xor_indices(self):
            value = aux.xor_indices(self.value)
            return Tensor(value)

        def im2col(self, h_filter, w_filter, strides, padding):
            i2c = im2col(
                self.value, h_filter, w_filter, strides=strides, padding=padding
            )
            return Tensor(i2c)

        def im2patches(self, patch_size, strides, padding, data_format="NCHW"):
            i2p = im2patches(
                self.value,
                patch_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )
            return Tensor(i2p)

        def patches2im(
            self,
            patch_size,
            strides,
            padding,
            img_size=None,
            consolidation="SUM",
            data_format="NCHW",
        ):
            p2i = patches2im(
                self.value,
                patch_size,
                strides=strides,
                padding=padding,
                img_size=img_size,
                consolidation=consolidation,
                data_format=data_format,
            )
            return Tensor(p2i)

        def conv2d(self, other, strides: list, padding: str = "SAME"):
            if EXPLICIT_MODULUS is not None:
                # TODO(Morten) any good reason this wasn't implemented for PrimeTensor?
                raise NotImplementedError()
            x, y = _lift(self, other)
            return Tensor(conv2d(x.value, y.value, strides=strides, padding=padding))

        def batch_to_space(self, block_shape, crops):
            value = tf.batch_to_space(self.value, block_shape, crops)
            return Tensor(value)

        def space_to_batch(self, block_shape, paddings):
            value = tf.space_to_batch(self.value, block_shape, paddings)
            return Tensor(value)

        def mod(self, k: int):
            value = self.value % k
            if EXPLICIT_MODULUS is not None:
                value %= EXPLICIT_MODULUS
            return Tensor(value)

        def transpose(self, perm):
            return Tensor(tf.transpose(self.value, perm))

        def strided_slice(self, args, kwargs):
            return Tensor(tf.strided_slice(self.value, *args, **kwargs))

        def gather(self, indices: list, axis: int = 0):
            return Tensor(tf.gather(self.value, indices, axis=axis))

        def split(self, num_split: Union[int, list], axis: int = 0):
            values = tf.split(self.value, num_split, axis=axis)
            return [Tensor(value) for value in values]

        def scatter_nd(self, indices, shape):
            value = tf.scatter_nd(indices, self.value, shape)
            return Tensor(value)

        def reverse(self, axis):
            value = tf.reverse(self.value, axis)
            return Tensor(value)

        def reshape(self, axes: Union[tf.Tensor, List[int]]):
            return Tensor(tf.reshape(self.value, axes))

        def negative(self):
            value = tf.negative(self.value)
            if EXPLICIT_MODULUS is not None:
                value %= EXPLICIT_MODULUS
            return Tensor(value)

        def reduce_sum(self, axis, keepdims=None):
            value = tf.reduce_sum(self.value, axis, keepdims)
            if EXPLICIT_MODULUS is not None:
                value %= EXPLICIT_MODULUS
            return Tensor(value)

        def cumsum(self, axis, exclusive, reverse):
            value = tf.cumsum(
                self.value, axis=axis, exclusive=exclusive, reverse=reverse
            )
            if EXPLICIT_MODULUS is not None:
                value %= EXPLICIT_MODULUS
            return Tensor(value)

        def reduce_max(self, axis=None, keepdims=False):
            value = tf.reduce_max(self.value, axis, keepdims)
            return Tensor(value)

        def equal_zero(self, factory=None):
            factory = factory or FACTORY
            return factory.tensor(
                tf.cast(tf.equal(self.value, 0), dtype=factory.native_type)
            )

        def equal(self, other, factory=None):
            x, y = _lift(self, other)
            factory = factory or FACTORY
            return factory.tensor(
                tf.cast(tf.equal(x.value, y.value), dtype=factory.native_type)
            )

        def truncate(self, amount, base=2):
            if base == 2:
                return self.right_shift(amount)
            factor = base**amount
            factor_inverse = inverse(factor, self.factory.modulus)
            return (self - (self % factor)) * factor_inverse

        def right_shift(self, bitlength):
            return Tensor(tf.bitwise.right_shift(self.value, bitlength))

        def expand_dims(self, axis: Optional[int] = None):
            return Tensor(tf.expand_dims(self.value, axis))

        def squeeze(self, axis: Optional[List[int]] = None):
            return Tensor(tf.squeeze(self.value, axis=axis))

        def cast(self, factory):
            return factory.tensor(self.value)

        def __or__(self, other):
            return self.bitwise_or(other)

        def bitwise_or(self, other):
            x, y = _lift(self, other)
            value = tf.bitwise.bitwise_or(x.value, y.value)
            return Tensor(value)

        def __xor__(self, other):
            return self.bitwise_xor(other)

        def bitwise_xor(self, other):
            x, y = _lift(self, other)
            value = tf.bitwise.bitwise_xor(x.value, y.value)
            return Tensor(value)

        def __and__(self, other):
            return self.bitwise_and(other)

        def bitwise_and(self, other):
            x, y = _lift(self, other)
            value = tf.bitwise.bitwise_and(x.value, y.value)
            return Tensor(value)

        def __invert__(self):
            return self.invert()

        def invert(self):
            value = tf.bitwise.invert(self.value)
            return Tensor(value)

        def __lshift__(self, bitlength):
            return self.left_shift(bitlength)

        def left_shift(self, bitlength):
            return Tensor(tf.bitwise.left_shift(self.value, bitlength))

        def __rshift__(self, bitlength):
            """
            Arithmetic shift.
            Please refer to `self.logical_rshift` for a logical right shift.
            """
            return self.right_shift(bitlength)

        def logical_rshift(self, bitlength):
            """Computes a bitshift to the right."""
            # There is some bug in TF when casting from int to uint: the uint result
            # becomes 0 so the following code does not work.
            # Bug report: https://github.com/tensorflow/tensorflow/issues/30215
            #
            # cast_map = {tf.int8: tf.uint8, tf.int16: tf.uint16,
            #             tf.int32: tf.uint32, tf.int64: tf.uint64}
            # x = tf.bitwise.right_shift(
            #     tf.cast(self.value, dtype=cast_map[NATIVE_TYPE]), bitlength)
            # x = tf.cast(x, NATIVE_TYPE)
            #
            # Instead, we have to do the following slightly more sophisticated stuff.
            if bitlength < 0:
                raise ValueError("Unsupported shift steps.")
            if bitlength == 0:
                return self
            total = NATIVE_TYPE.size * 8
            mask = ~((-1) << (total - bitlength))
            x = tf.bitwise.right_shift(self.value, bitlength)
            x = tf.bitwise.bitwise_and(x, mask)
            return Tensor(x)

        def bit_reverse(self):
            value = aux.bit_reverse(self.value)
            return Tensor(value)

    class Constant(Tensor, AbstractConstant):
        """Native Constant class."""

        def __init__(self, constant_value: tf.Tensor) -> None:
            super(Constant, self).__init__(constant_value)

        def __repr__(self) -> str:
            return "Constant(shape={})".format(self.shape)

    class Variable(Tensor, AbstractVariable):
        """Native Variable class."""

        def __init__(self, variable_value: tf.Variable) -> None:
            # TODO(zjn) need a better implementation to update Tensor's `_value`
            # after variable has been assigned
            # also for case bool factory and int100 factory
            super(Variable, self).__init__(variable_value.read_value())
            self.variable = variable_value

        def __repr__(self) -> str:
            return "Variable(shape={})".format(self.shape)

        def assign(self, value: Union[Tensor, np.ndarray]) -> None:
            if isinstance(value, Tensor):
                return self.variable.assign(value.value)
            elif isinstance(value, np.ndarray):
                return self.variable.assign(value)
            else:
                raise TypeError("Don't know how to handle {}".format(type(value)))

        def read_value(self) -> Tensor:
            return Tensor(self.variable.read_value())

    return FACTORY
