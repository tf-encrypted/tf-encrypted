from __future__ import absolute_import

from typing import Optional, Tuple
import math
import random
import sys

import numpy as np
import tensorflow as tf

from .odd_tensor import oddInt64factory
from ..protocol import memoize, nodes
from ...protocol.pond import (
    Pond, PondTensor, PondPublicTensor, PondPrivateTensor, PondMaskedTensor, _type
)
from ...tensor import native_factory, int64factory
from ...tensor.factory import AbstractFactory, AbstractTensor
from ...player import Player
from ...config import get_config


_thismodule = sys.modules[__name__]


class SecureNN(Pond):
    """
    SecureNN(server_0, server_1, server_2, prime_factory, odd_factory, **kwargs)

    Implementation of SecureNN from `Wagh et al <https://eprint.iacr.org/2018/442/>`_.
    """

    def __init__(self,
                 server_0: Optional[Player] = None,
                 server_1: Optional[Player] = None,
                 server_2: Optional[Player] = None,
                 tensor_factory: Optional[AbstractFactory] = None,
                 prime_factory: Optional[AbstractFactory] = None,
                 odd_factory: Optional[AbstractFactory] = None,
                 **kwargs) -> None:
        server_0 = server_0 or get_config().get_player('server0')
        server_1 = server_1 or get_config().get_player('server1')
        server_2 = server_2 \
            or get_config().get_player('server2') \
            or get_config().get_player('crypto-producer')

        assert server_0 is not None
        assert server_1 is not None
        assert server_2 is not None

        super(SecureNN, self).__init__(
            server_0=server_0,
            server_1=server_1,
            crypto_producer=server_2,
            tensor_factory=tensor_factory,
            **kwargs
        )
        self.server_2 = server_2

        if odd_factory is None:
            if self.tensor_factory is int64factory:
                odd_factory = oddInt64factory
            else:
                odd_factory = self.tensor_factory

        if prime_factory is None:
            prime = 107
            assert prime > math.ceil(math.log2(self.tensor_factory.modulus))
            prime_factory = native_factory(self.tensor_factory.native_type, prime)

        self.prime_factory = prime_factory
        self.odd_factory = odd_factory
        assert self.prime_factory.native_type == self.tensor_factory.native_type
        assert self.odd_factory.native_type == self.tensor_factory.native_type

    @memoize
    def bitwise_not(self, x: PondTensor) -> PondTensor:
        """
        bitwise_not(x) -> PondTensor

        Computes the bitwise `NOT` of the input, i.e. :math:`f(x) = 1 - x`.

        :param PondTensor x: Input tensor.
        """
        assert not x.is_scaled, "Input is not supposed to be scaled"
        with tf.name_scope('bitwise_not'):
            return self.sub(1, x)

    @memoize
    def bitwise_and(self, x: 'PondTensor', y: 'PondTensor') -> 'PondTensor':
        """
        bitwise_and(x, y) -> PondTensor

        Computes the bitwise `AND` of the given inputs, :math:`f(x,y) = xy`.

        :param PondTensor x: Input tensor.
        :param PondTensor y: Input tensor.
        """
        assert not x.is_scaled, "Input is not supposed to be scaled"
        assert not y.is_scaled, "Input is not supposed to be scaled"
        with tf.name_scope('bitwise_and'):
            return x * y

    @memoize
    def bitwise_or(self, x: 'PondTensor', y: 'PondTensor') -> 'PondTensor':
        """
        bitwise_or(x, y) -> PondTensor

        Computes the bitwise `OR` of the given inputs, :math:`f(x,y) = x + y - xy`.

        :param PondTensor x: Input tensor.
        :param PondTensor y: Input tensor.
        """
        assert not x.is_scaled, "Input is not supposed to be scaled"
        assert not y.is_scaled, "Input is not supposed to be scaled"
        with tf.name_scope('bitwise_or'):
            return x + y - self.bitwise_and(x, y)

    @memoize
    def bitwise_xor(self, x: 'PondTensor', y: 'PondTensor') -> 'PondTensor':
        """
        bitwise_xor(x, y) -> PondTensor

        Compute the bitwise `XOR` of the given inputs, :math:`f(x,y) = x + y - 2xy`

        :param PondTensor x: Input tensor.
        :param PondTensor y: Input tensor.
        """
        assert not x.is_scaled, "Input is not supposed to be scaled"
        assert not y.is_scaled, "Input is not supposed to be scaled"
        with tf.name_scope('bitwise_xor'):
            return x + y - self.bitwise_and(x, y) * 2

    @memoize
    def msb(self, x: 'PondTensor') -> 'PondTensor':
        """
        msb(x) -> PondTensor

        Computes the most significant bit of the provided tensor.

        :param PondTensor x: The tensor to take the most significant bit of
        """
        with tf.name_scope('msb'):
            # when the modulus is odd msb reduces to lsb via x -> 2*x
            x = self.cast_backing(x, self.odd_factory)
            return self.lsb(x + x)

    @memoize
    def lsb(self, x: PondTensor) -> PondTensor:
        """
        lsb(x) -> PondTensor

        Computes the least significant bit of the provided tensor.

        :param PondTensor x: The tensor to take the least significant bit of.
        """
        return self.dispatch('lsb', x, container=_thismodule)

    @memoize
    def bits(self,
             x: PondPublicTensor,
             factory: Optional[AbstractFactory] = None) -> 'PondPublicTensor':
        """
        bits(x, factory) -> PondPublicTensor

        Convert a fixed-point precision tensor into its bitwise representation.

        :param PondPublicTensor x: A fixed-point tensor to extract into a bitwise representation.
        """
        return self.dispatch('bits', x, container=_thismodule, factory=factory)

    @memoize
    def is_negative(self, x: PondTensor) -> PondTensor:
        """
        Returns :math:`x < 0`.

        .. code-block:: python

            >>> negative([-1, 0, 1])
            [1, 0, 0]

        :param PondTensor x: The tensor to check.
        """
        with tf.name_scope('is_negative'):
            # NOTE MSB is 1 iff xi < 0
            return self.msb(x)

    @memoize
    def non_negative(self, x: PondTensor) -> PondTensor:
        """
        non_negative(x) -> PondTensor

        Returns :math:`x >= 0`.

        .. code-block:: python

            >>> non_negative([-1, 0, 1])
            [0, 1, 1]

        Note this is the derivative of the ReLU function.

        :param PondTensor x: The tensor to check.
        """
        with tf.name_scope('non_negative'):
            return self.bitwise_not(self.msb(x))

    @memoize
    def less(self, x: PondTensor, y: PondTensor) -> PondTensor:
        """
        less(x, y) -> PondTensor

        Returns :math:`x < y`.

        .. code-block:: python

            >>> less([1,2,3], [0,1,5])
            [0, 0, 1]

        :param PondTensor x: The tensor to check.
        :param PondTensor y: The tensor to check against.
        """
        with tf.name_scope('less'):
            return self.is_negative(x - y)

    @memoize
    def less_equal(self, x: PondTensor, y: PondTensor) -> PondTensor:
        """
        less_equal(x, y) -> PondTensor

        Returns :math:`x <= y`.

        .. code-block:: python

            >>> less_equal([1,2,3], [0,1,3])
            [0, 0, 1]

        :param PondTensor x: The tensor to check.
        :param PondTensor y: The tensor to check against.
        """
        with tf.name_scope('less_equal'):
            return self.bitwise_not(self.greater(x, y))

    @memoize
    def greater(self, x: PondTensor, y: PondTensor) -> PondTensor:
        """
        greater(x, y) -> PondTensor

        Returns :math:`x > y`.

        .. code-block:: python

            >>> greater([1,2,3], [0,1,5])
            [1, 1, 0]

        :param PondTensor x: The tensor to check.
        :param PondTensor y: The tensor to check against.
        """
        with tf.name_scope('greater'):
            return self.is_negative(y - x)

    @memoize
    def greater_equal(self, x: PondTensor, y: PondTensor) -> PondTensor:
        """
        greater_equal(x, y) -> PondTensor

        Returns :math:`x >= y`.

        .. code-block:: python

            >>> greater_equal([1,2,3], [0,1,3])
            [1, 1, 1]

        :param PondTensor x: The tensor to check.
        :param PondTensor y: The tensor to check against.
        """
        with tf.name_scope('greater_equal'):
            return self.bitwise_not(self.less(x, y))

    @memoize
    def select(self, choice_bit: PondTensor, x: PondTensor, y: PondTensor) -> PondTensor:
        """
        select(choice_bit, x, y) -> PondTensor

        The `select` protocol from Wagh et al.  Secretly selects and returns elements from two candidate tensors.

        .. code-block:: python

            >>> option_x = [10, 20, 30, 40]
            >>> option_y = [1, 2, 3, 4]
            >>> select(choice_bit=1, x=option_x, y=option_y)
            [1, 2, 3, 4]
            >>> select(choice_bit=[0,1,0,1], x=option_x, y=option_y)
            [10, 2, 30, 4]

        `NOTE:` Inputs to this function in real use will not look like above.
        In practice these will be secret shares.

        :param PondTensor choice_bit: The bits representing which tensor to choose.
            If `choice_bit = 0` then choose elements from `x`, otherwise choose from `y`.
        :param PondTensor x: Candidate tensor 0.
        :param PondTensor y: Candidate tensor 1.
        """  # noqa:E501

        # TODO[Morten] optimize select when choice_bit is a public tensor

        # TODO[Morten]
        # these assertions should ideally be enabled but requires lifting to be
        # applied to the inputs first; make sure that's fixed during refactoring
        #
        # assert x.backing_dtype == y.backing_dtype
        # assert x.is_scaled == y.is_scaled
        # assert not choice_bit.is_scaled

        with tf.name_scope('select'):
            return (y - x) * choice_bit + x

    @memoize
    def equal_zero(self, x, dtype: Optional[AbstractFactory] = None):
        """
        equal_zero(x, dtype) -> PondTensor

        Evaluates the Boolean expression :math:`x = 0`.

        .. code-block:: python

            >>> equal_zero([1,0,1])
            [0, 1, 0]

        :param PondTensor x: The tensor to evaluate.
        :param AbstractFactory dtype: An optional tensor factory, defaults to dtype of `x`.
        """
        return self.dispatch('equal_zero', x, container=_thismodule, dtype=dtype)

    @memoize
    def relu(self, x):
        """
        relu(x) -> PondTensor

        Returns the exact `ReLU` by computing `ReLU(x) = x * nonnegative(x)`.

        .. code-block:: python

            >>> relu([-12, -3, 1, 3, 3])
            [0, 0, 1, 3, 3]

        :param PondTensor x: Input tensor.
        """
        with tf.name_scope('relu'):
            drelu = self.non_negative(x)
            return drelu * x

    def maxpool2d(self, x, pool_size, strides, padding):
        """
        maxpool2d(x, pool_size, strides, padding) -> PondTensor

        Performs a `MaxPooling2d` operation on `x`.

        :param PondTensor x: Input tensor.
        :param List[int] pool_size: The size of the pool.
        :param List[int] strides: A list describing how to stride over the convolution.
        :param str padding: Which type of padding to use ("SAME" or "VALID").
        """
        node_key = ('maxpool2d', x, tuple(pool_size), tuple(strides), padding)
        z = nodes.get(node_key, None)

        if z is not None:
            return z

        dispatch = {
            PondPublicTensor: _maxpool2d_public,
            PondPrivateTensor: _maxpool2d_private,
            PondMaskedTensor: _maxpool2d_masked,
        }

        func = dispatch.get(_type(x), None)
        if func is None:
            raise TypeError("Don't know how to avgpool2d {}".format(type(x)))

        z = func(self, x, pool_size, strides, padding)
        nodes[node_key] = z

        return z

    @memoize
    def maximum(self, x, y):
        """
        maximum(x, y) -> PondTensor

        Computes :math:`max(x,y)`.

        Returns the greater value of each tensor per index.

        .. code-block:: python

            >>> maximum([10, 20, 30], [11, 19, 31])
            [11, 20, 31]

        :param PondTensor x: Input tensor.
        :param PondTensor y: Input tensor.
        """
        with tf.name_scope('maximum'):
            indices_of_maximum = self.greater(x, y)
            return self.select(indices_of_maximum, y, x)

    @memoize
    def reduce_max(self, x, axis=0):
        """
        reduce_max(x, axis) -> PondTensor

        Find the max value along an axis.

        .. code-block:: python

            >>> reduce_max([[10, 20, 30], [11, 13, 12], [15, 16, 17]], axis=0)
            [[30], [13], [17]]

        :See: tf.reduce_max
        :param PondTensor x: Input tensor.
        :param int axis: The tensor axis to reduce along.
        :rtype: PondTensor
        :returns: A new tensor with the specified axis reduced to the max value in that axis.
        """
        with tf.name_scope('reduce_max'):

            def build_comparison_tree(ts):
                if len(ts) == 1:
                    return ts[0]
                halfway = len(ts) // 2
                ts_left, ts_right = ts[:halfway], ts[halfway:]
                maximum_left = build_comparison_tree(ts_left)
                maximum_right = build_comparison_tree(ts_right)
                return self.maximum(maximum_left, maximum_right)

            tensors = self.split(x, int(x.shape[axis]), axis=axis)
            maximum = build_comparison_tree(tensors)
            return self.squeeze(maximum, axis=(axis,))

    @memoize
    def argmax(self, x, axis=0):
        """
        argmax(x, axis) -> PondTensor

        Find the index of the max value along an axis.

        .. code-block:: python

            >>> argmax([[10, 20, 30], [11, 13, 12], [15, 16, 17]], axis=0)
            [[2], [1], [2]]

        :See: tf.argmax
        :param PondTensor x: Input tensor.
        :param int axis: The tensor axis to reduce along.
        :rtype: PondTensor
        :returns: A new tensor with the indices of the max values along specified axis.
        """
        with tf.name_scope('argmax'):

            def build_comparison_tree(tensors, indices):
                assert len(tensors) == len(indices)
                if len(indices) == 1:
                    return tensors[0], indices[0]

                halfway = len(tensors) // 2
                tensors_left, tensors_right = tensors[:halfway], tensors[halfway:]
                indices_left, indices_right = indices[:halfway], indices[halfway:]

                maximum_left, argmax_left = build_comparison_tree(tensors_left, indices_left)
                maximum_right, argmax_right = build_comparison_tree(tensors_right, indices_right)

                # compute binary tensor indicating which side is greater
                greater = self.greater(maximum_left, maximum_right)

                # use above binary tensor to select maximum and argmax
                maximum = self.select(greater, maximum_right, maximum_left)
                argmax = self.select(greater, argmax_right, argmax_left)

                return maximum, argmax

            tensors = self.split(x, int(x.shape[axis]), axis=axis)
            indices = [
                self.define_constant(np.array([i]))
                for i, _ in enumerate(tensors)
            ]

            with tf.name_scope('comparison-tree'):
                maximum, argmax = build_comparison_tree(tensors, indices)

            maximum = self.squeeze(maximum, axis=(axis,))
            argmax = self.squeeze(argmax, axis=(axis,))
            return argmax

    @memoize
    def cast_backing(self, x, backing_dtype):
        return self.dispatch("cast_backing", x, backing_dtype, container=_thismodule)


def _bits_public(prot, x: PondPublicTensor,
                 factory: Optional[AbstractFactory] = None) -> PondPublicTensor:

    factory = factory or prot.tensor_factory

    with tf.name_scope('bits'):

        x_on_0, x_on_1 = x.unwrapped

        with tf.device(prot.server_0.device_name):
            bits_on_0 = x_on_0.bits(factory)

        with tf.device(prot.server_1.device_name):
            bits_on_1 = x_on_1.bits(factory)

        return PondPublicTensor(prot, bits_on_0, bits_on_1, False)


def _lsb_public(prot, x: PondPublicTensor):
    # TODO[Morten]
    # we could call through and ask the underlying dtype for its lsb instead as there might
    # be more efficient ways of getting it for some types (ie without getting all bits)
    x_bits = prot.bits(x)
    x_lsb = x_bits[..., 0]
    return x_lsb


def _lsb_private(prot, x: PondPrivateTensor):

    # TODO[Morten] in the refactor these could be type parameters
    odd_dtype = x.backing_dtype
    out_dtype = prot.tensor_factory
    prime_dtype = prot.prime_factory

    assert odd_dtype.modulus % 2 == 1
    assert x.backing_dtype == odd_dtype  # needed for security because of `r` masking

    with tf.name_scope('lsb'):

        with tf.name_scope('blind'):

            # ask server2 to generate r mask and its bits
            with tf.device(prot.server_2.device_name):
                r0 = odd_dtype.sample_uniform(x.shape)
                r1 = odd_dtype.sample_uniform(x.shape)
                r = PondPrivateTensor(prot, r0, r1, False)

                r_raw = r0 + r1
                rbits_raw = r_raw.bits(factory=prime_dtype)
                rbits = prot._share_and_wrap(rbits_raw, False)

                # TODO[Morten] once .bits() is cached then call .lsb() here instead
                rlsb_raw = rbits_raw[..., 0].cast(out_dtype)
                rlsb = prot._share_and_wrap(rlsb_raw, False)

            # blind and reveal
            c = (x + r).reveal()
            c = prot.cast_backing(c, out_dtype)
            c.is_scaled = False

        with tf.name_scope('compare'):

            # ask either server0 and server1 to generate beta (distributing load)
            server = random.choice([prot.server_0, prot.server_1])
            with tf.device(server.device_name):
                beta_raw = prime_dtype.sample_bits(x.shape)
                beta = PondPublicTensor(prot, beta_raw, beta_raw, is_scaled=False)

            greater_xor_beta = _private_compare(prot, rbits, c, beta)
            clsb = prot.lsb(c)

        with tf.name_scope('unblind'):
            gamma = prot.bitwise_xor(greater_xor_beta, prot.cast_backing(beta, out_dtype))
            delta = prot.bitwise_xor(rlsb, clsb)
            alpha = prot.bitwise_xor(gamma, delta)

        assert alpha.backing_dtype is out_dtype
        return alpha


def _lsb_masked(prot, x: PondMaskedTensor):
    return prot.lsb(x.unmasked)


def _private_compare(prot, x_bits: PondPrivateTensor, r: PondPublicTensor, beta: PondPublicTensor):
    # TODO[Morten] no need to check this (should be free)
    assert x_bits.backing_dtype == prot.prime_factory
    assert r.backing_dtype == prot.tensor_factory

    out_shape = r.shape
    out_dtype = r.backing_dtype
    prime_dtype = x_bits.backing_dtype
    bit_length = x_bits.shape[-1]

    assert r.shape == out_shape
    assert r.backing_dtype == out_dtype
    assert not r.is_scaled

    assert x_bits.shape[:-1] == out_shape
    assert x_bits.backing_dtype == prime_dtype
    assert not x_bits.is_scaled

    assert beta.shape == out_shape
    assert beta.backing_dtype == prime_dtype
    assert not beta.is_scaled

    with tf.name_scope('private_compare'):

        with tf.name_scope('bit_comparisons'):

            # use either r or t = r + 1 according to beta
            s = prot.select(prot.cast_backing(beta, out_dtype), r, r + 1)
            s_bits = prot.bits(s, factory=prime_dtype)
            assert s_bits.shape[-1] == bit_length

            # compute w_sum
            w_bits = prot.bitwise_xor(x_bits, s_bits)
            w_sum = prot.cumsum(w_bits, axis=-1, reverse=True, exclusive=True)
            assert w_sum.backing_dtype == prime_dtype

            # compute c, ignoring edge cases at first
            sign = prot.select(beta, 1, -1)
            sign = prot.expand_dims(sign, axis=-1)
            c_except_edge_case = (s_bits - x_bits) * sign + 1 + w_sum

            assert c_except_edge_case.backing_dtype == prime_dtype

        with tf.name_scope('edge_cases'):

            # adjust for edge cases, i.e. where beta is 1 and s is zero (meaning r was -1)

            # identify edge cases
            edge_cases = prot.bitwise_and(
                beta,
                prot.equal_zero(s, prime_dtype)
            )
            edge_cases = prot.expand_dims(edge_cases, axis=-1)

            # tensor for edge cases: one zero and the rest ones
            c_edge_case_raw = prime_dtype.tensor(tf.constant([0] + [1] * (bit_length - 1),
                                                             dtype=prime_dtype.native_type,
                                                             shape=(1, bit_length)))
            c_edge_case = prot._share_and_wrap(c_edge_case_raw, False)

            c = prot.select(
                edge_cases,
                c_except_edge_case,
                c_edge_case
            )  # type: PondPrivateTensor
            assert c.backing_dtype == prime_dtype

        with tf.name_scope('zero_search'):

            # generate multiplicative mask to hide non-zero values
            with tf.device(prot.server_0.device_name):
                mask_raw = prime_dtype.sample_uniform(c.shape, minval=1)
                mask = PondPublicTensor(prot, mask_raw, mask_raw, False)

            # mask non-zero values; this is safe when we're in a prime dtype (since it's a field)
            c_masked = c * mask
            assert c_masked.backing_dtype == prime_dtype

            # TODO[Morten] permute

            # reconstruct masked values on server 2 to find entries with zeros
            with tf.device(prot.server_2.device_name):
                d = prot._reconstruct(*c_masked.unwrapped)
                # find all zero entries
                zeros = d.equal_zero(out_dtype)
                # for each bit sequence, determine whether it has one or no zero in it
                rows_with_zeros = zeros.reduce_sum(axis=-1, keepdims=False)
                # reshare result
                result = prot._share_and_wrap(rows_with_zeros, False)

        assert result.backing_dtype == out_dtype
        return result


def _equal_zero_public(prot,
                       x: PondPublicTensor,
                       dtype: Optional[AbstractFactory] = None) -> PondPublicTensor:

    with tf.name_scope('equal_zero'):

        x_on_0, x_on_1 = x.unwrapped

        with tf.device(prot.server_0.device_name):
            equal_zero_on_0 = x_on_0.equal_zero(dtype)

        with tf.device(prot.server_1.device_name):
            equal_zero_on_1 = x_on_1.equal_zero(dtype)

        return PondPublicTensor(prot, equal_zero_on_0, equal_zero_on_1, False)


#
# max pooling helpers
#

def _im2col(prot: Pond,
            x: PondTensor,
            pool_size: Tuple[int, int],
            strides: Tuple[int, int],
            padding: str) -> Tuple[AbstractTensor, AbstractTensor]:

    x_on_0, x_on_1 = x.unwrapped
    batch, channels, height, width = x.shape

    if padding == "SAME":
        out_height = math.ceil(int(height) / strides[0])
        out_width = math.ceil(int(width) / strides[1])
    else:
        out_height = math.ceil((int(height) - pool_size[0] + 1) / strides[0])
        out_width = math.ceil((int(width) - pool_size[1] + 1) / strides[1])

    batch, channels, height, width = x.shape
    pool_height, pool_width = pool_size

    with tf.device(prot.server_0.device_name):
        x_split = x_on_0.reshape((batch * channels, 1, height, width))
        y_on_0 = x_split.im2col(pool_height, pool_width, padding, strides[0])

    with tf.device(prot.server_1.device_name):
        x_split = x_on_1.reshape((batch * channels, 1, height, width))
        y_on_1 = x_split.im2col(pool_height, pool_width, padding, strides[0])

    return y_on_0, y_on_1, [out_height, out_width, int(batch), int(channels)]


def _maxpool2d_public(prot: Pond,
                      x: PondPublicTensor,
                      pool_size: Tuple[int, int],
                      strides: Tuple[int, int],
                      padding: str) -> PondPublicTensor:

    with tf.name_scope('maxpool2d'):
        y_on_0, y_on_1, reshape_to = _im2col(prot, x, pool_size, strides, padding)
        im2col = PondPublicTensor(prot, y_on_0, y_on_1, x.is_scaled)
        max = im2col.reduce_max(axis=0)
        result = max.reshape(reshape_to).transpose([2, 3, 0, 1])
        return result


def _maxpool2d_private(prot: Pond,
                       x: PondPrivateTensor,
                       pool_size: Tuple[int, int],
                       strides: Tuple[int, int],
                       padding: str) -> PondPrivateTensor:

    with tf.name_scope('maxpool2d'):
        y_on_0, y_on_1, reshape_to = _im2col(prot, x, pool_size, strides, padding)
        im2col = PondPrivateTensor(prot, y_on_0, y_on_1, x.is_scaled)
        max = im2col.reduce_max(axis=0)
        result = max.reshape(reshape_to).transpose([2, 3, 0, 1])
        return result


def _maxpool2d_masked(prot: Pond,
                      x: PondMaskedTensor,
                      pool_size: Tuple[int, int],
                      strides: Tuple[int, int],
                      padding: str) -> PondPrivateTensor:

    with tf.name_scope('maxpool2d'):
        return prot.maxpool2d(x.unwrapped, pool_size, strides, padding)


#
# cast helpers
#


def _cast_backing_public(prot: Pond, x: PondPublicTensor, backing_dtype) -> PondPublicTensor:

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope("cast_backing"):

        with tf.device(prot.server_0.device_name):
            y_on_0 = x_on_0.cast(backing_dtype)

        with tf.device(prot.server_1.device_name):
            y_on_1 = x_on_1.cast(backing_dtype)

        return PondPublicTensor(prot, y_on_0, y_on_1, x.is_scaled)


def _cast_backing_private(prot: Pond, x: PondPrivateTensor, backing_dtype) -> PondPrivateTensor:

    # TODO[Morten]
    # this method is risky as it differs from what the user might expect, which would normally
    # require more advanced convertion protocols accounting for wrap-around etc;
    # for this reason we should consider hiding it during refactoring

    x0, x1 = x.unwrapped

    with tf.name_scope("cast_backing"):

        with tf.device(prot.server_0.device_name):
            y0 = x0.cast(backing_dtype)

        with tf.device(prot.server_1.device_name):
            y1 = x1.cast(backing_dtype)

        return PondPrivateTensor(prot, y0, y1, x.is_scaled)
