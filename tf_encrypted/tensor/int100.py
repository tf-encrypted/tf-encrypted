"""High-precision tensors that implement a fixed-point representation with
an array of backing tensors with lower precision conforming to the Chinese
remainder theorem.

Currently, we only use the CRT for an int100 tensor, although other
high-precision tensor types are possible."""
from __future__ import absolute_import
from typing import Union, Optional, List, Tuple
from functools import reduce, partial
import abc
import math

import numpy as np
import tensorflow as tf

from .helpers import prod, inverse
from .factory import (
    AbstractFactory, AbstractTensor, AbstractVariable,
    AbstractConstant, AbstractPlaceholder
)
from .shared import binarize, conv2d, im2col
from ..operations import secure_random


def crt_factory(INT_TYPE, MODULI):  # pylint: disable=invalid-name
  """Chinese remainder theorem tensor factory."""

  # pylint: disable=invalid-name
  MATMUL_THRESHOLD = 1024

  MODULUS = prod(MODULI)
  BITSIZE = math.ceil(math.log2(MODULUS))
  # pylint: enable=invalid-name

  # make sure we have room for lazy reductions:
  # - 1 multiplication followed by 1024 additions
  for mi in MODULI:
    assert 2 * math.log2(mi) + math.log2(1024) < math.log2(INT_TYPE.max)

  #
  # methods benefitting from precomputation
  #

  def gen_crt_recombine_lagrange():

    # precomputation
    n = [MODULUS // mi for mi in MODULI]
    lambdas = [ni * inverse(ni, mi) % MODULUS for ni, mi in zip(n, MODULI)]

    def crt_recombine_lagrange(x):

      with tf.name_scope('crt_recombine_lagrange'):
        res = sum(xi * li for xi, li in zip(x, lambdas)) % MODULUS
        res = res.astype(object)
        return res

    return crt_recombine_lagrange

  def gen_crt_recombine_explicit():

    # precomputation
    q = [inverse(MODULUS // mi, mi) for mi in MODULI]

    def crt_recombine_explicit(x, bound):

      big_b = MODULUS % bound
      b = [(MODULUS // mi) % bound for mi in MODULI]

      with tf.name_scope('crt_recombine_explicit'):

        if isinstance(x[0], np.ndarray):
          # backed by np.ndarray
          t = [(xi * qi) % mi for xi, qi, mi in zip(x, q, MODULI)]
          alpha = np.round(
              np.sum(
                  [ti.astype(float) / mi for ti, mi in zip(t, MODULI)],
                  axis=0
              ))
          u = np.sum([ti * bi for ti, bi in zip(t, b)],
                     axis=0).astype(np.int64)
          v = alpha.astype(np.int64) * big_b
          w = u - v
          res = w % bound
          res = res.astype(np.int32)
          return res

        if isinstance(x[0], tf.Tensor):
          # backed by tf.Tensor
          t = [(xi * qi) % mi for xi, qi, mi in zip(x, q, MODULI)]
          alpha = tf.round(
              tf.reduce_sum(
                  [tf.cast(ti, tf.float32) / mi for ti, mi in zip(t, MODULI)],
                  axis=0
              ))
          u = tf.cast(tf.reduce_sum(
              [ti * bi for ti, bi in zip(t, b)], axis=0), tf.int64)
          v = tf.cast(alpha, tf.int64) * big_b
          w = u - v
          res = w % bound
          res = tf.cast(res, INT_TYPE)
          return res

        raise TypeError("Don't know how to recombine {}".format(type(x[0])))

    return crt_recombine_explicit

  def gen_crt_mod():

    # outer precomputation
    q = [inverse(MODULUS // mi, mi) for mi in MODULI]

    def crt_mod(x, k):
      assert isinstance(k, int), type(k)

      # inner precomputations
      big_b = MODULUS % k
      b = [(MODULUS // mi) % k for mi in MODULI]

      with tf.name_scope('crt_mod'):
        t = [(xi * qi) % mi for xi, qi, mi in zip(x, q, MODULI)]
        alpha = tf.round(
            tf.reduce_sum(
                [tf.cast(ti, tf.float32) / mi for ti, mi in zip(t, MODULI)],
                axis=0
            )
        )
        u = tf.reduce_sum([ti * bi for ti, bi in zip(t, b)], axis=0)
        v = tf.cast(alpha, INT_TYPE) * big_b
        w = u - v
        return w % k

    return crt_mod

  crt_recombine_lagrange = gen_crt_recombine_lagrange()
  crt_recombine_explicit = gen_crt_recombine_explicit()
  crt_mod = gen_crt_mod()

  #
  # methods used in more than one place
  #

  def _crt_decompose(x):
    return [x % mi for mi in MODULI]

  def _crt_add(x, y):
    return [(xi + yi) % mi for xi, yi, mi in zip(x, y, MODULI)]

  def _crt_sub(x, y):
    return [(xi - yi) % mi for xi, yi, mi in zip(x, y, MODULI)]

  def _crt_mul(x, y):
    return [(xi * yi) % mi for xi, yi, mi in zip(x, y, MODULI)]

  def _crt_matmul(x, y):
    return [tf.matmul(xi, yi) % mi for xi, yi, mi in zip(x, y, MODULI)]

  def _construct_backing_from_chunks(chunk_sizes, chunk_values):
    backing = _crt_decompose(0)
    for chunk_size, chunk_value in zip(chunk_sizes, chunk_values):
      scale = 2**chunk_size
      backing = _crt_add(
          _crt_mul(backing, _crt_decompose(scale)),
          _crt_decompose(chunk_value)
      )
    return backing

  class Factory(AbstractFactory):
    """CRT tensor factory."""

    def zero(self):
      backing = [tf.constant(0, dtype=INT_TYPE)] * len(MODULI)
      return DenseTensor(backing)

    def one(self):
      backing = [tf.constant(1, dtype=INT_TYPE)] * len(MODULI)
      return DenseTensor(backing)

    def sample_uniform(self,
                       shape,
                       minval: Optional[int] = None,
                       maxval: Optional[int] = None):
      assert minval is None
      assert maxval is None

      if secure_random.supports_seeded_randomness():
        seeds = [secure_random.secure_seed() for _ in MODULI]
        return UniformTensor(shape, seeds)

      if secure_random.supports_secure_randomness():
        backing = [secure_random.random_uniform(shape,
                                                minval=0,
                                                maxval=mi,
                                                dtype=INT_TYPE)
                   for mi in MODULI]
        return DenseTensor(backing)

      backing = [tf.random_uniform(shape,
                                   minval=0,
                                   maxval=mi,
                                   dtype=INT_TYPE)
                 for mi in MODULI]
      return DenseTensor(backing)

    def sample_bounded(self,
                       shape,
                       bitlength: int):

      # TODO[Morten] bump to full range once signed numbers is settled (change minval etc)
      chunk_max_bitlength = 30

      q, r = bitlength // chunk_max_bitlength, bitlength % chunk_max_bitlength
      chunk_sizes = [chunk_max_bitlength] * q + ([r] if r > 0 else [])

      if secure_random.supports_seeded_randomness():
        seeds = [secure_random.secure_seed() for _ in chunk_sizes]
        return BoundedTensor(shape=shape,
                             seeds=seeds,
                             chunk_sizes=chunk_sizes)

      if secure_random.supports_secure_randomness():
        sampler = secure_random.random_uniform
      else:
        sampler = tf.random_uniform

      chunk_values = [sampler(shape=shape,
                              minval=0,
                              maxval=2**chunk_size,
                              dtype=INT_TYPE)
                      for chunk_size in chunk_sizes]
      backing = _construct_backing_from_chunks(chunk_sizes, chunk_values)
      return DenseTensor(backing)

    def stack(self, xs: list, axis: int = 0):
      assert all(isinstance(x, Tensor) for x in xs)
      backing = [tf.stack([x.backing[i] for x in xs], axis=axis)
                 for i in range(len(xs[0].backing))]
      return DenseTensor(backing)

    def concat(self, xs: list, axis: int = 0):
      assert all(isinstance(x, Tensor) for x in xs)
      backing = [tf.concat([x.backing[i] for x in xs], axis=axis)
                 for i in range(len(xs[0].backing))]
      return DenseTensor(backing)

    def tensor(self, value):

      if isinstance(value, tf.Tensor):
        backing = [tf.cast(component, dtype=INT_TYPE)
                   for component in _crt_decompose(value)]
        return DenseTensor(backing)

      if isinstance(value, np.ndarray):
        backing = [tf.convert_to_tensor(component, dtype=INT_TYPE)
                   for component in _crt_decompose(value)]
        return DenseTensor(backing)

      raise TypeError(("Don't know how to handle ",
                       "{}".format(type(value))))

    def constant(self, value):

      if isinstance(value, np.ndarray):
        backing = [tf.constant(v, dtype=INT_TYPE)
                   for v in _crt_decompose(value)]
        return Constant(backing)

      raise TypeError(("Don't know how to handle ",
                       "{}".format(type(value))))

    def variable(self, initial_value):

      if isinstance(initial_value, (tf.Tensor, np.ndarray)):
        return Variable(_crt_decompose(initial_value))

      if isinstance(initial_value, Tensor):
        return Variable(initial_value.backing)

      raise TypeError(("Don't know how to handle ",
                       "{}".format(type(initial_value))))

    def placeholder(self, shape):
      return Placeholder(shape)

    @property
    def min(self):
      return 0

    @property
    def max(self):
      return MODULUS

    @property
    def modulus(self) -> int:
      return MODULUS

    @property
    def native_type(self):
      return INT_TYPE

  master_factory = Factory()

  def _lift(x, y) -> Tuple['Tensor', 'Tensor']:

    if isinstance(x, Tensor) and isinstance(y, Tensor):
      return x, y

    if isinstance(x, Tensor):

      if isinstance(y, int):
        return x, x.factory.tensor(np.array([y]))

    if isinstance(y, Tensor):

      if isinstance(x, int):
        return y.factory.tensor(np.array([x])), y

    raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))

  class Tensor(AbstractTensor):
    """Base class for other CRT tensor classes."""

    @abc.abstractproperty
    @property
    def backing(self):
      pass

    @abc.abstractproperty
    @property
    def shape(self):
      pass

    @property
    def modulus(self):
      return MODULUS

    @property
    def factory(self):
      return master_factory

    def to_native(self) -> Union[tf.Tensor, np.ndarray]:
      return crt_recombine_explicit(self.backing, 2**32)

    def bits(
        self,
        factory: Optional[AbstractFactory] = None,
        ensure_positive_interpretation: bool = False
    ) -> AbstractTensor:
      """Convert to a pure bits representation."""

      factory = factory or self.factory

      with tf.name_scope('to_bits'):

        # we will extract the bits in chunks of 16 as that's reasonable for the explicit CRT
        max_chunk_bitsize = 16
        q, r = BITSIZE // max_chunk_bitsize, BITSIZE % max_chunk_bitsize
        chunk_bitsizes = [max_chunk_bitsize] * q + ([r] if r > 0 else [])
        chunks_modulus = [2**bitsize for bitsize in chunk_bitsizes]

        remaining = self

        if ensure_positive_interpretation:

          # To get the right bit pattern for negative numbers we need to apply a correction
          # to the first chunk. Unfortunately, this isn't known until all bits have been
          # extracted and hence we extract bits both with and without the correction and
          # select afterwards. Although these two versions could be computed independently
          # we here combine them into a single tensor to keep the graph smaller.

          shape = self.shape.as_list()
          shape_value = [1] + shape
          shape_correction = [2] + [1] * len(shape)

          # this means that chunk[0] is uncorrected and chunk[1] is corrected
          correction_raw = [0, self.modulus % chunks_modulus[0]]
          correction = tf.constant(correction_raw,
                                   shape=shape_correction,
                                   dtype=self.factory.native_type)

          remaining = remaining.reshape(shape_value)

        # extract chunks
        chunks = []
        apply_correction = ensure_positive_interpretation
        for chunk_modulus in chunks_modulus:

          # extract chunk from remaining
          chunk = crt_mod(remaining.backing, chunk_modulus)

          # apply correction only to the first chunk
          if apply_correction:
            chunk = (chunk + correction) % chunk_modulus
            apply_correction = False

          # save for below
          chunks.append(chunk)

          # perform right shift on remaining
          shifted = (remaining - master_factory.tensor(chunk))
          remaining = shifted * inverse(chunk_modulus, self.modulus)

        if ensure_positive_interpretation:
          # pick between corrected and uncorrected based on MSB
          msb = chunks[-1][0] >= (chunks_modulus[-1]) // 2
          chunks = [
              tf.where(
                  msb,
                  chunk[1],  # corrected
                  chunk[0],  # uncorrected
              )
              for chunk in chunks
          ]

        # extract bits from chunks
        chunks_bits = [
            binarize(chunk, chunk_bitsize)
            for chunk, chunk_bitsize in zip(chunks, chunk_bitsizes)
        ]

        # combine bits of chunks
        bits = tf.concat(chunks_bits, axis=-1)

        return factory.tensor(bits)

    def to_bigint(self) -> np.ndarray:
      return crt_recombine_lagrange(self.backing)

    def __getitem__(self, slc):
      return DenseTensor([x[slc] for x in self.backing])

    def __repr__(self) -> str:
      return 'Tensor({})'.format(self.shape)

    def __add__(self, other):
      x, y = _lift(self, other)
      return x.add(y)

    def __radd__(self, other):
      x, y = _lift(self, other)
      return x.add(y)

    def __sub__(self, other):
      x, y = _lift(self, other)
      return x.sub(y)

    def __rsub__(self, other):
      x, y = _lift(self, other)
      return x.sub(y)

    def __mul__(self, other):
      x, y = _lift(self, other)
      return x.mul(y)

    def __rmul__(self, other):
      x, y = _lift(self, other)
      return x.mul(y)

    def __mod__(self, k: int):
      return self.mod(k)

    def add(self, other):
      x, y = _lift(self, other)
      return DenseTensor(_crt_add(x.backing, y.backing))

    def sub(self, other):
      x, y = _lift(self, other)
      return DenseTensor(_crt_sub(x.backing, y.backing))

    def mul(self, other):
      x, y = _lift(self, other)
      return DenseTensor(_crt_mul(x.backing, y.backing))

    def matmul(self, other):
      """Matmul with other."""
      x, y = _lift(self, other)

      if x.shape[1] <= MATMUL_THRESHOLD:
        # perform matmul directly (we have enough room)
        z_backing = _crt_matmul(x.backing, y.backing)
        return DenseTensor(z_backing)

      # we need to split the tensors, process independently, and then recombine

      with tf.name_scope('split'):
        z_split = []

        num_columns = int(x.backing[0].shape[1])
        num_split = int(math.ceil(num_columns / MATMUL_THRESHOLD))
        for i in range(num_split):

          left = i * MATMUL_THRESHOLD
          right = (i + 1) * MATMUL_THRESHOLD

          inner_x = []  # type: List[Union[tf.Tensor, np.ndarray]]
          inner_y = []  # type: List[Union[tf.Tensor, np.ndarray]]

          for xi, yi in zip(x.backing, y.backing):
            inner_x.append(xi[:, left:right])
            inner_y.append(yi[left:right, :])

          z_split.append((inner_x, inner_y))

      with tf.name_scope('recombine'):
        split_products = [_crt_matmul(xi, yi) for xi, yi in z_split]
        z_backing = reduce(_crt_add, split_products)

      return DenseTensor(z_backing)

    def mod(self, k: int):
      backing = _crt_decompose(crt_mod(self.backing, k))
      return DenseTensor(backing)

    def reduce_sum(self, axis, keepdims=None):
      with tf.name_scope('crt_reduce_sum'):
        backing = [tf.reduce_sum(xi, axis, keepdims) % mi
                   for xi, mi in zip(self.backing, MODULI)]
        return DenseTensor(backing)

    def cumsum(self, axis, exclusive, reverse):
      with tf.name_scope('crt_cumsum'):
        backing = [tf.cumsum(xi,
                             axis=axis,
                             exclusive=exclusive,
                             reverse=reverse) % mi
                   for xi, mi in zip(self.backing, MODULI)]
        return DenseTensor(backing)

    def equal_zero(self, factory=None):
      """Check equality with zero."""
      factory = factory or master_factory

      with tf.name_scope('crt_equal_zero'):
        zeros = [tf.cast(tf.equal(xi, 0), factory.native_type)
                 for xi in self.backing]
        number_of_zeros = tf.reduce_sum(zeros, axis=0)
        backing = tf.equal(number_of_zeros, len(MODULI))
        all_zeros = tf.cast(backing, factory.native_type)

      return factory.tensor(all_zeros)

    def equal(self, other, factory=None):
      """Check equality with other."""
      x, y = _lift(self, other)
      factory = factory or x.factory

      with tf.name_scope('crt_equal'):
        matches = [tf.cast(tf.equal(xi, yi), factory.native_type)
                   for xi, yi in zip(x.backing, y.backing)]
        number_of_matches = tf.reduce_sum(matches, axis=0)
        backing = tf.equal(number_of_matches, len(MODULI))
        all_matches = tf.cast(backing, factory.native_type)

      return factory.tensor(all_matches)

    def im2col(self, h_filter: int, w_filter: int, padding: str, stride: int):
      with tf.name_scope('crt_im2col'):
        backing = [im2col(xi,
                          h_filter=h_filter,
                          w_filter=w_filter,
                          padding=padding,
                          stride=stride)
                   for xi in self.backing]
        return DenseTensor(backing)

    def conv2d(self, other, stride: int, padding: str = 'SAME'):
      x, y = _lift(self, other)
      return conv2d(x, y, stride, padding)  # type: ignore

    def batch_to_space_nd(self, block_shape, crops):
      with tf.name_scope("crt_batch_to_space_nd"):
        backing = [tf.batch_to_space_nd(xi,
                                        block_shape=block_shape,
                                        crops=crops)
                   for xi in self.backing]
        return DenseTensor(backing)

    def space_to_batch_nd(self, block_shape, paddings):
      with tf.name_scope("crt_space_to_batch_nd"):
        backing = [tf.space_to_batch_nd(xi,
                                        block_shape=block_shape,
                                        paddings=paddings)
                   for xi in self.backing]
        return DenseTensor(backing)

    def transpose(self, perm):
      backing = [tf.transpose(xi, perm=perm) for xi in self.backing]
      return DenseTensor(backing)

    def strided_slice(self, args, kwargs):
      backing = [tf.strided_slice(xi, *args, **kwargs) for xi in self.backing]
      return DenseTensor(backing)

    def gather(self, indices: list, axis: int = 0):
      backing = [tf.gather(xi, indices, axis=axis) for xi in self.backing]
      return DenseTensor(backing)

    def split(self, num_split: Union[int, list], axis: int = 0):
      backings = zip(*[tf.split(xi, num_split, axis=axis)
                       for xi in self.backing])
      return [DenseTensor(backing) for backing in backings]

    def reshape(self, axes: List[int]):
      backing = [tf.reshape(xi, axes) for xi in self.backing]
      return DenseTensor(backing)

    def negative(self):
      backing = [tf.negative(xi) % mi for xi, mi in zip(self.backing, MODULI)]
      return DenseTensor(backing)

    def expand_dims(self, axis: Optional[int] = None):
      backing = [tf.expand_dims(xi, axis) for xi in self.backing]
      return DenseTensor(backing)

    def squeeze(self, axis: Optional[List[int]] = None):
      backing = [tf.squeeze(xi, axis=axis) for xi in self.backing]
      return DenseTensor(backing)

    def truncate(self, amount, base=2):
      factor = base**amount
      factor_inverse = inverse(factor, MODULUS)
      return (self - (self % factor)) * factor_inverse

    def right_shift(self, bitlength):
      return self.truncate(bitlength, 2)

    def cast(self, factory):
      # NOTE(Morten) could add more convertion options
      if factory is self.factory:
        return self
      raise ValueError("Don't know how to cast into {}".format(factory))

  class DenseTensor(Tensor):
    """Public CRT Tensor class."""

    def __init__(self, backing):
      self._backing = backing

    @property
    def shape(self):
      return self._backing[0].shape

    @property
    def backing(self):
      return self._backing

  class UniformTensor(Tensor):
    """Class representing a uniform-random, lazily sampled tensor.

    Lazy sampling optimizes communication by sending seeds in place of
    fully-expanded tensors."""

    def __init__(self, shape, seeds):
      self._seeds = seeds
      self._shape = shape

    @property
    def shape(self):
      return self._shape

    @property
    def backing(self):
      with tf.name_scope('expand-seed'):
        return [secure_random.seeded_random_uniform(self._shape,
                                                    minval=0,
                                                    maxval=mi,
                                                    seed=seed,
                                                    dtype=INT_TYPE)
                for (mi, seed) in zip(MODULI, self._seeds)]

  class BoundedTensor(Tensor):
    """CRT bounded-randomness tensor."""

    def __init__(self, shape, seeds, chunk_sizes):
      self._shape = shape
      self._seeds = seeds
      self._chunk_sizes = chunk_sizes

    @property
    def shape(self):
      return self._shape

    @property
    def backing(self):
      with tf.name_scope('expand-seed'):
        sampler = partial(secure_random.seeded_random_uniform,
                          self._shape,
                          minval=0,
                          dtype=INT_TYPE)
        zipped = zip(self._chunk_sizes, self._seeds)
        chunk_values = [sampler(maxval=2 ** chunk_size, seed=seed_value)
                        for chunk_size, seed_value in zipped]
        return _construct_backing_from_chunks(self._chunk_sizes, chunk_values)

  class Constant(DenseTensor, AbstractConstant):
    """CRT Constant class."""

    def __init__(self, backing) -> None:
      assert all(isinstance(component, tf.Tensor) for component in backing)
      super(Constant, self).__init__(backing)

    def __repr__(self) -> str:
      return 'Constant({})'.format(self.shape)

  class Placeholder(DenseTensor, AbstractPlaceholder):
    """CRT Placeholder class."""

    def __init__(self, shape) -> None:
      self.placeholders = [tf.placeholder(INT_TYPE, shape=shape)
                           for _ in MODULI]
      super(Placeholder, self).__init__(self.placeholders)

    def __repr__(self):
      return 'Placeholder({})'.format(self.shape)

    def feed(self, value):
      assert isinstance(value, np.ndarray), type(value)
      backing = _crt_decompose(value)
      return {
          p: v for p, v in zip(self.placeholders, backing)
      }

  class Variable(DenseTensor, AbstractVariable):
    """CRT Variable class."""

    def __init__(self, initial_backing) -> None:
      self.variables = [tf.Variable(val,
                                    dtype=INT_TYPE,
                                    trainable=False)
                        for val in initial_backing]
      self.initializer = tf.group(*[var.initializer for var in self.variables])
      backing = [var.read_value() for var in self.variables]
      super(Variable, self).__init__(backing)

    def __repr__(self):
      return 'Variable({})'.format(self.shape)

    def assign_from_native(self, value: np.ndarray):
      assert isinstance(value, np.ndarray), type(value)
      return self.assign_from_same(master_factory.tensor(value))

    def assign_from_same(self, value: Tensor):
      assert isinstance(value, Tensor), type(value)
      assign_ops = [tf.assign(xi, vi).op
                    for xi, vi in zip(self.variables, value.backing)]
      return tf.group(*assign_ops)

  return master_factory


#
# 32 bit CRT
# - we need this to do matmul as int32 is the only supported type for that
# - tried tf.float64 but didn't work out of the box
# - 10 components for modulus ~100 bits
#

int100factory = crt_factory(
    INT_TYPE=tf.int32,
    MODULI=[1201, 1433, 1217, 1237, 1321, 1103, 1129, 1367, 1093, 1039],
)
