"""Native tensors and their factory.

These use TensorFlow's native dtypes tf.int64 for the given float
encoding being used (fixed-point, etc.)."""
import numpy as np
import tensorflow as tf

from typing import Union, List, Tuple, Optional

from .shared import binarize, conv2d, im2col, im2patches, patches2im
from .factory import (AbstractFactory, AbstractTensor, AbstractVariable,
                      AbstractConstant)
from .helpers import inverse
from ..operations import tf_i128
from ..operations import secure_random

from tensorflow.python.framework.tensor_shape import TensorShape


def int128_factory():  # pylint: disable=invalid-name
    """Constructs the 128 bits tensor Factory."""

    class Factory(AbstractFactory):
        """128 bits tensor factory."""

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
                encode_value = tf_i128.encode(scaled_value)
                return tf.convert_to_tensor(encode_value, dtype=self.native_type)
            elif isinstance(scaled_value, np.ndarray):
                encode_value = tf_i128.encode(scaled_value)
                return tf.convert_to_tensor(encode_value, dtype=self.native_type)
            elif isinstance(scaled_value, tf.Tensor):
                return tf_i128.to_i128(scaled_value)
            else:
                raise TypeError(
                    "Don't know how to handle {}".format(type(scaled_value))
                )
        
        def _decode(self, encode_value):
            if isinstance(encode_value, tf.Tensor):
                return tf_i128.from_i128(encode_value)
            else:
                raise TypeError(
                    "Don't know how to handle {}".format(type(encode_value))
                )

        @property
        def min(self):
            return -2**127

        @property
        def max(self):
            return 2**127-1

        @property
        def min_as_int64(self):
            return tf.constant([0, tf.int64.min], dtype=tf.int64)

        @property
        def max_as_int64(self):
            return tf.constant([-1, tf.int64.max], dtype=tf.int64)

        @property
        def modulus(self) -> int:
            return self.max - self.min + 1

        @property
        def native_type(self):
            return tf.int64

        @property
        def native_size(self):
            # Number of native_type elements needed to represent one composed element
            return 2

        @property
        def nbits(self):
            return self.native_size * self.native_type.size * 8

        def sample_uniform(self,
                           shape,
                           minval=None,
                           maxval=None):
            minval = self.min_as_int64 if minval is None else minval
            maxval = self.max_as_int64 if maxval is None else maxval

            seed = secure_random.secure_seed()
            return Tensor(secure_random.i128_seeded_random_uniform(
                shape=shape,
                seed=seed,
                minval=minval,
                maxval=maxval,
                ))

        def sample_seeded_uniform(self,
                                  shape,
                                  seed,
                                  minval=None,
                                  maxval=None):
            minval = self.min_as_int64 if minval is None else minval
            maxval = self.max_as_int64 if maxval is None else maxval

            # Don't use UniformTensor for lazy sampling here, because the `seed` might be something (e.g., key) we
            # want to protect, and we cannot send it to another party
            value = secure_random.i128_seeded_random_uniform(
                shape=shape,
                seed=seed,
                minval=minval,
                maxval=maxval
            )
            return Tensor(value)

        def sample_seeded_bounded(self,
                                  shape,
                                  seed,
                                  bitlength: int):
            assert bitlength <= 128

            # Don't use UniformTensor for lazy sampling here, because the `seed` might be something (e.g., key) we
            # want to protect, and we cannot send it to another party
            value = secure_random.i128_seeded_random_uniform(
                shape=shape,
                seed=seed,
                minval=self.min_as_int64,
                maxval=self.max_as_int64,
            )

            r = Tensor(value).logical_rshift(128 - bitlength)
            return r

        def sample_bounded(self, shape, bitlength: int):
            seed = secure_random.secure_seed()
            return self.sample_seeded_bounded(shape, seed, bitlength)

        def sample_bits(self, shape):
            return self.sample_bounded(shape, bitlength=1)

        def stack(self, xs: list, axis: int = 0):
            assert all(isinstance(x, Tensor) for x in xs)
            axis = _lift_axis(axis, len(xs[0].shape))

            value = tf.stack([x.value for x in xs], axis=axis)
            return Tensor(value)

        def concat(self, xs: list, axis: int):
            assert all(isinstance(x, Tensor) for x in xs)
            axis = _lift_axis(axis, len(xs[0].shape))

            value = tf.concat([x.value for x in xs], axis=axis)
            return Tensor(value)

        def ones(self, shape):
            if not isinstance(shape, (list, tuple, TensorShape)):
                raise TypeError("shape must be a list or tuple of integers")
            lo = tf.ones(shape, self.native_type)
            hi = tf.zeros(shape, self.native_type)
            value = tf.stack([lo, hi], axis=len(shape))
            return Tensor(value)

        def zeros(self, shape):
            if not isinstance(shape, (list, tuple, TensorShape)):
                raise TypeError("shape must be a list or tuple of integers")
            if isinstance(shape, (list, tuple)):
                new_shape = tf.TensorShape(list(shape) + [self.native_size])
            else:
                new_shape = tf.TensorShape(shape.as_list() + [self.native_size])
            return Tensor(tf.zeros(new_shape, self.native_type))

        def ones_like(self, x: 'Tensor'):
            return self.ones(x.shape)

        def zeros_like(self, x: 'Tensor'):
            return self.zeros(x.shape)

        def where(self, condition, x, y):
            if not isinstance(condition, tf.Tensor):
                msg = "Don't know how to handle `condition` of type {}"
                raise TypeError(msg.format(type(condition)))
            # Replicate the condition so that it is compatible with the shape of x and y
            condition = tf.stack([condition, condition], axis=len(condition.shape))
            value = tf.where(condition, x.value, y.value)
            return Tensor(value)

        def tile(self, input, multiples):
            if isinstance(multiples, tf.TensorShape):
                multiples = multiples.as_list()

            if isinstance(multiples, tuple):
                multiples = multiples + (1,)
            elif isinstance(multiples, list):
                multiples = multiples + [1]
            elif isinstance(multiples, tf.Tensor):
                multiples = tf.concat([multiples, tf.constant([1])], axis=0)
            else:
                raise TypeError("multiples must be a tuple, a list or a tf.Tensor")
            return Tensor(tf.tile(input.value, multiples))
        
        def scatter_nd(self, indices, updates, shape):
            if isinstance(indices, Tensor):
                indices = tf.gather(indices.value, 0, axis=-1)
            
            r0 = tf.scatter_nd(indices, tf.gather(updates.value, 0, axis=-1), shape)
            r1 = tf.scatter_nd(indices, tf.gather(updates.value, 1, axis=-1), shape)
            return Tensor(tf.stack([r0, r1], axis=-1))


    def _lift(x, y) -> Tuple['Tensor', 'Tensor']:

        if isinstance(x, Tensor) and isinstance(y, Tensor):
            return x, y

        if isinstance(x, Tensor):
            return x, x.factory.tensor(np.array(y))

        if isinstance(y, Tensor):
            return y.factory.tensor(np.array(x)), y

        raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))
    
    def _lift_axis(axis, dims):
        if isinstance(axis, (list, tuple)):
            new_axis = []
            for a in axis:
                if a == -1:
                    new_axis.append(dims - 1)
                else:
                    assert a < dims, "# dimensions: {}, got axis: {}".format(dims, axis)
                    new_axis.append(a)
            return new_axis
        if axis == -1:
            return dims - 1
        if axis is not None:
            assert axis < dims, "# dimensions: {}, got axis: {}".format(dims, axis)
        return axis

    FACTORY = Factory()  # pylint: disable=invalid-name

    class Tensor(AbstractTensor):
        """Base class for other native tensor classes."""

        def __init__(self, value) -> None:
            self._value = value

        @property
        def value(self):
            return self._value

        @property
        def shape(self):
            # Remove the last dimension
            return self._value.shape[0:len(self._value.shape)-1]

        @property
        def native_shape(self):
            return self._value.shape

        def identity(self):
            value = tf.identity(self.value)
            return Tensor(value)

        def to_native(self) -> tf.Tensor:
            return self.factory._decode(self.value)
        
        def bits(self, factory=None, bitsize=128) -> AbstractTensor:
            factory = factory or FACTORY
            if bitsize <= 64:
                t = tf.gather(self.value, indices=0, axis=-1)
                return factory.tensor(binarize(t, bitsize))
            else:
                t0 = tf.gather(self.value, 0, axis=-1)
                t1 = tf.gather(self.value, 1, axis=-1)
                t0 = binarize(t0, bitsize)
                t1 = binarize(t1, bitsize)
                return factory.tensor(tf.stack([t0, t1], axis=-1))

        def __repr__(self) -> str:
            return '{}(shape={})'.format(type(self), self.shape)

        @property
        def factory(self):
            return FACTORY
        
        @property
        def device(self):
            return self._value.device

        @property
        def dtype(self):
            return self.factory.native_type

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
            return self.negative()

        def __getitem__(self, slc):
            return Tensor(self.value[slc])

        def add(self, other):
            x, y = _lift(self, other)
            value = tf_i128.add(x.value, y.value)
            return Tensor(value)

        def sub(self, other):
            x, y = _lift(self, other)
            value = tf_i128.sub(x.value, y.value)
            return Tensor(value)

        def mul(self, other):
            x, y = _lift(self, other)
            value = tf_i128.mul(x.value, y.value)
            return Tensor(value)

        def matmul(self, other):
            x, y = _lift(self, other)
            value = tf_i128.matmul(x.value, y.value)
            return Tensor(value)

        def bit_reverse(self):
            value = tf_i128.i128_bit_reverse(self.value)
            return Tensor(value)

        def bit_gather(self, start, stride):
            value = tf_i128.i128_bit_gather(self.value, start, stride)
            return Tensor(value)
        
        def bit_split_and_gather(self, stride):
            value = tf_i128.i128_bit_split_and_gather(self.value, stride)
            return Tensor(value)

        def xor_indices(self):
            value = tf_i128.i128_xor_indices(self.value)
            return Tensor(value)

        def im2col(self, h_filter, w_filter, strides, padding):
            i2c = im2col(
                self, h_filter, w_filter, strides=strides, padding=padding
            )
            return i2c
        
        def im2patches(self, patch_size, strides, padding, data_format="NCHW"):
            patch0 = im2patches(
                tf.gather(self.value, 0, axis=-1),
                patch_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )
            patch1 = im2patches(
                tf.gather(self.value, 1, axis=-1),
                patch_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )
            patches = tf.stack([patch0, patch1], axis=-1)
            return Tensor(patches)
        
        def patches2im(
            self,
            patch_size,
            strides,
            padding,
            img_size=None,
            consolidation="SUM",
            data_format="NCHW",
        ):
            p2i0 = patches2im(
                tf.gather(self.value, 0, axis=-1),
                patch_size,
                strides=strides,
                padding=padding,
                img_size=img_size,
                consolidation=consolidation,
                data_format=data_format,
            )
            p2i1 = patches2im(
                tf.gather(self.value, 1, axis=-1),
                patch_size,
                strides=strides,
                padding=padding,
                img_size=img_size,
                consolidation=consolidation,
                data_format=data_format,
            )
            p2is = tf.stack([p2i0, p2i1], axis=-1)
            return Tensor(p2is)

        def conv2d(self, other, stride: int, padding: str = 'SAME'):
            x, y = _lift(self, other)
            return conv2d(x, y, stride, padding)

        def batch_to_space(self, block_shape, crops):
            value = tf.batch_to_space(self.value, block_shape, crops)
            return Tensor(value)

        def space_to_batch(self, block_shape, paddings):
            value = tf.space_to_batch(self.value, block_shape, paddings)
            return Tensor(value)

        def transpose(self, perm):
            n_dims = len(self.shape)
            if perm is None:
                perm = list(range(n_dims-1, -1, -1)) + [n_dims]
            else:
                perm = list(perm) + [n_dims]
            return Tensor(tf.transpose(self.value, perm))

        def strided_slice(self, args, kwargs):
            return Tensor(tf.strided_slice(self.value, *args, **kwargs))

        def gather(self, indices: list, axis: int = 0):
            if isinstance(indices, Tensor):
                indices = tf.gather(indices.value, 0, axis=-1)

            axis = _lift_axis(axis, len(self.shape))
            return Tensor(tf.gather(self.value, indices, axis=axis))

        def split(self, num_split: Union[int, list], axis: int = 0):
            axis = _lift_axis(axis, len(self.shape))
            values = tf.split(self.value, num_split, axis=axis)
            return [Tensor(value) for value in values]
        
        def scatter_nd(self, indices, shape):
            if isinstance(indices, Tensor):
                indices = tf.gather(indices.value, 0, axis=-1)

            r0 = tf.scatter_nd(indices, tf.gather(self.value, 0, axis=-1), shape)
            r1 = tf.scatter_nd(indices, tf.gather(self.value, 1, axis=-1), shape)
            return Tensor(tf.stack([r0, r1], axis=-1))
        
        def reverse(self, axis):
            axis = _lift_axis(axis, len(self.shape))
            value = tf.reverse(self.value, axis)
            return Tensor(value)

        def reshape(self, axes: Union[tf.Tensor, List[int], Tuple[int]]):
            if isinstance(axes, tuple):
                axes = axes + (self.factory.native_size,)
            elif isinstance(axes, list):
                axes = axes + [self.factory.native_size]
            elif isinstance(axes, tf.TensorShape):
                axes = axes.as_list() + [self.factory.native_size]
            else:
                raise TypeError("axes has unexpected type: {}".format(type(axes)))
            return Tensor(tf.reshape(self.value, axes))

        def negative(self):
            value = tf_i128.negate(self.value)
            return Tensor(value)

        def reduce_sum(self, axis, keepdims=None):
            axis = _lift_axis(axis, len(self.shape))
            value = tf_i128.reduce_sum(self.value, axis, keepdims)
            return Tensor(value)

        def equal(self, other):
            x, y = _lift(self, other)
            return tf_i128.equal(x.value, y.value)

        def truncate(self, amount, base=2):
            if base == 2:
                return self.right_shift(amount)
            factor = base**amount
            factor_inverse = inverse(factor, self.factory.modulus)
            return (self - (self % factor)) * factor_inverse

        def expand_dims(self, axis):
            axis = _lift_axis(axis, len(self.shape)+1)
            return Tensor(tf.expand_dims(self.value, axis))

        def squeeze(self, axis: Optional[List[int]] = None):
            axis = _lift_axis(axis, len(self.shape))
            return Tensor(tf.squeeze(self.value, axis=axis))

        def cast(self, factory):
            if factory == self.factory:
                return self
            elif factory.native_type == tf.bool:
                split_bool = tf.cast(self.value, tf.bool)
                merge_bool = tf.math.logical_or(
                        tf.gather(split_bool, 0, axis=-1),
                        tf.gather(split_bool, 1, axis=-1)
                )
                return factory.tensor(merge_bool)
            else:
                return factory.tensor(tf.gather(self.value, 0, axis=-1))

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
            # Because "and" is a keyword in Python, the naming "and_" follows the way how Python handles this:
            # https://docs.python.org/3.4/library/operator.html
            x, y = _lift(self, other)
            value = tf.bitwise.bitwise_and(x.value, y.value)
            return Tensor(value)

        def __invert__(self):
            return self.invert()

        def invert(self):
            value = tf.bitwise.invert(self.value)
            return Tensor(value)

        def __lshift__(self, bitlength):
            return self.lshift(bitlength)

        def lshift(self, bitlength):
            return Tensor(tf_i128.left_shift(self.value, bitlength))

        def __rshift__(self, bitlength):
          """
          Arithmetic shift.
          Please refer to `self.logical_rshift` if a logical right shift is desired.
          """
          return self.right_shift(bitlength)
        
        def right_shift(self, bitlength):
            return Tensor(tf_i128.right_shift(self.value, bitlength))

        def logical_rshift(self, bitlength):
            return Tensor(tf_i128.logic_right_shift(self.value, bitlength))

    class Constant(Tensor, AbstractConstant):
        """Native Constant class."""

        def __init__(self, constant: tf.Tensor) -> None:
            assert isinstance(constant, tf.Tensor)
            super(Constant, self).__init__(constant)

        def __repr__(self) -> str:
            return 'Constant(shape={})'.format(self.shape)

    class Variable(Tensor, AbstractVariable):
        """Native Variable class."""

        def __init__(self, variable_value: tf.Variable) -> None:
            # TODO(zjn) need a better implementation to update Tensor's `_value`
            # after variable has been assigned
            # also for case bool factory and int100 factory
            super(Variable, self).__init__(variable_value.read_value())
            self.variable = variable_value

        def __repr__(self) -> str:
            return 'Variable(shape={})'.format(self.shape)

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
