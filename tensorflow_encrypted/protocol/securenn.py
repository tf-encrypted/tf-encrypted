from __future__ import absolute_import
from typing import Optional, Tuple
import sys
import math

import tensorflow as tf

from .protocol import memoize, nodes
from ..protocol.pond import (
    Pond, PondTensor, PondPublicTensor, PondPrivateTensor, PondMaskedTensor, _type
)
from ..tensor.prime import PrimeFactory
from ..tensor.factory import AbstractFactory, AbstractTensor
from ..player import Player
from ..config import get_config


_thismodule = sys.modules[__name__]


class SecureNN(Pond):

    def __init__(
        self,
        server_0: Optional[Player] = None,
        server_1: Optional[Player] = None,
        server_2: Optional[Player] = None,
        prime_factory: Optional[AbstractFactory] = None,
        odd_factory: Optional[AbstractFactory] = None,
        **kwargs
    ) -> None:
        server_0 = server_0 or get_config().get_player('server0')
        server_1 = server_1 or get_config().get_player('server1')
        server_2 = server_2 or get_config().get_player('crypto_producer')  # TODO[Morten] use `server2` as key

        super(SecureNN, self).__init__(
            server_0=server_0,
            server_1=server_1,
            crypto_producer=server_2,
            **kwargs
        )
        self.server_2 = server_2
        self.prime_factory = prime_factory or PrimeFactory(107)
        self.odd_factory = odd_factory or self.tensor_factory

    @memoize
    def bitwise_not(self, x: PondTensor) -> PondTensor:
        assert not x.is_scaled, "Input is not supposed to be scaled"
        with tf.name_scope('bitwise_not'):
            return self.sub(1, x)

    @memoize
    def bitwise_and(self, x: PondTensor, y: PondTensor) -> PondTensor:
        assert not x.is_scaled, "Input is not supposed to be scaled"
        assert not y.is_scaled, "Input is not supposed to be scaled"
        with tf.name_scope('bitwise_and'):
            return x * y

    @memoize
    def bitwise_or(self, x: PondTensor, y: PondTensor) -> PondTensor:
        assert not x.is_scaled, "Input is not supposed to be scaled"
        assert not y.is_scaled, "Input is not supposed to be scaled"
        with tf.name_scope('bitwise_or'):
            return x + y - self.bitwise_and(x, y)

    @memoize
    def bitwise_xor(self, x: PondTensor, y: PondTensor) -> PondTensor:
        assert not x.is_scaled, "Input is not supposed to be scaled"
        assert not y.is_scaled, "Input is not supposed to be scaled"
        with tf.name_scope('bitwise_xor'):
            return x + y - self.bitwise_and(x, y) * 2

    @memoize
    def msb(self, x: PondTensor) -> PondTensor:
        # NOTE when the modulus is odd then msb reduces to lsb via x -> 2*x
        if x.backing_dtype.modulus % 2 != 1:
            # NOTE: this is currently only for use with an odd-modulus CRTTensor
            #       NativeTensor will use an even modulus and will require share_convert
            raise Exception('SecureNN protocol assumes a ring of odd cardinality, ' +
                            'but it was initialized with an even one.')
        return self.lsb(x * 2)

    @memoize
    def lsb(self, x: PondTensor) -> PondTensor:
        return self.dispatch('lsb', x, container=_thismodule)

    @memoize
    def bits(self, x: PondTensor, factory: Optional[AbstractFactory]=None) -> 'PondTensor':
        return self.dispatch('bits', x, container=_thismodule, factory=factory)

    @memoize
    def negative(self, x: PondTensor) -> PondTensor:
        with tf.name_scope('negative'):
            # NOTE MSB is 1 iff xi < 0
            return self.msb(x)

    @memoize
    def non_negative(self, x: PondTensor) -> PondTensor:
        with tf.name_scope('non_negative'):
            return self.bitwise_not(self.msb(x))

    @memoize
    def less(self, x: PondTensor, y: PondTensor) -> PondTensor:
        with tf.name_scope('less'):
            return self.negative(x - y)

    @memoize
    def less_equal(self, x: PondTensor, y: PondTensor) -> PondTensor:
        with tf.name_scope('less_equal'):
            return self.bitwise_not(self.greater(x, y))

    @memoize
    def greater(self, x: PondTensor, y: PondTensor) -> PondTensor:
        with tf.name_scope('greater'):
            return self.negative(y - x)

    @memoize
    def greater_equal(self, x: PondTensor, y: PondTensor) -> PondTensor:
        with tf.name_scope('greater_equal'):
            return self.bitwise_not(self.less(x, y))

    @memoize
    def select(self, choice_bit: PondTensor, x: PondTensor, y: PondTensor) -> PondTensor:
        with tf.name_scope('select'):
            return (y - x) * choice_bit + x

    @memoize
    def equal_zero(self, x, out_dtype: Optional[AbstractFactory]=None):
        return self.dispatch('equal_zero', x, container=_thismodule, out_dtype=out_dtype)

    def share_convert(self, x):
        raise NotImplementedError

    def divide(self, x, y):
        raise NotImplementedError

    @memoize
    def relu(self, x):
        with tf.name_scope('relu'):
            drelu = self.non_negative(x)
            return drelu * x

    def maxpool2d(self, x, pool_size, strides, padding):
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
        with tf.name_scope('maximum'):
            indices_of_maximum = self.greater(x, y)
            return self.select(indices_of_maximum, y, x)

    @memoize
    def reduce_max(self, x, axis=0):
        with tf.name_scope('reduce_max'):

            def build_comparison_tree(ts):
                assert len(ts) > 0
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

    def dmax_pool_efficient(self, x):
        raise NotImplementedError


def _bits_public(prot, x: PondPublicTensor, factory: Optional[AbstractFactory]=None) -> PondPublicTensor:

    factory = factory or prot.tensor_factory

    with tf.name_scope('bits'):

        x_on_0, x_on_1 = x.unwrapped

        with tf.device(prot.server_0.device_name):
            bits_on_0 = x_on_0.to_bits(factory)

        with tf.device(prot.server_1.device_name):
            bits_on_1 = x_on_1.to_bits(factory)

        return PondPublicTensor(prot, bits_on_0, bits_on_1, False)


def _lsb_private(prot, y: PondPrivateTensor):

    with tf.name_scope('lsb'):

        with tf.name_scope('lsb_mask'):

            with tf.device(prot.server_2.device_name):
                x_raw = y.backing_dtype.sample_uniform(y.shape)
                xbits_raw = x_raw.to_bits(factory=prot.prime_factory)
                xlsb_raw = xbits_raw[..., 0].cast(y.backing_dtype)

                x = prot._share_and_wrap(x_raw, False)
                xbits = prot._share_and_wrap(xbits_raw, False)
                xlsb = prot._share_and_wrap(xlsb_raw, False)

            with tf.device(prot.server_0.device_name):
                # TODO[Morten] pull this out as a separate `sample_bits` method on tensors (optimized for bits only)
                beta_raw = prot.prime_factory.sample_bounded(y.shape, 1)
                beta = PondPublicTensor(prot, beta_raw, beta_raw, is_scaled=False)

        with tf.name_scope('lsb_compare'):
            r = (y + x).reveal()
            rbits = prot.bits(r)
            rlsb = rbits[..., 0]
            bp = _private_compare(prot, xbits, r, beta)

        with tf.name_scope('lsb_combine'):
            gamma = prot.bitwise_xor(bp, beta.cast_backing(prot.tensor_factory))
            delta = prot.bitwise_xor(xlsb, rlsb)
            alpha = prot.bitwise_xor(gamma, delta)
            assert alpha.backing_dtype is y.backing_dtype

        return alpha


def _lsb_masked(prot, x: PondMaskedTensor):
    return prot.lsb(x.unmasked)


def _private_compare(prot, x_bits: PondPrivateTensor, r: PondPublicTensor, beta: PondPublicTensor):
    # TODO[Morten] no need to check this (should be free)
    assert r.backing_dtype == prot.tensor_factory
    assert x_bits.backing_dtype == prot.prime_factory

    out_shape = r.shape
    out_dtype = r.backing_dtype
    prime_dtype = x_bits.backing_dtype
    bit_length = x_bits.shape[-1]

    assert r.shape == out_shape
    assert r.backing_dtype == out_dtype
    assert x_bits.shape[:-1] == out_shape
    assert x_bits.backing_dtype == prime_dtype
    assert beta.shape == out_shape
    assert beta.backing_dtype == prime_dtype

    with tf.name_scope('private_compare'):

        with tf.name_scope('bit_comparisons'):

            # use either r or t = r + 1 according to beta
            s = prot.select(beta.cast_backing(r.backing_dtype), r, r + 1)
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

            edge_cases = prot.bitwise_and(
                beta,
                prot.equal_zero(s, prime_dtype)
            )
            edge_cases = prot.expand_dims(edge_cases, axis=-1)
            c_edge_case_raw = prime_dtype.tensor(tf.constant([0] + [1] * (bit_length - 1), dtype=tf.int32, shape=(1, bit_length)))
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

            # mask non-zero values; this is safe when we're in a field
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


def _equal_zero_public(prot, x: PondPublicTensor, out_dtype: Optional[AbstractFactory]=None) -> PondPublicTensor:

    with tf.name_scope('equal_zero'):

        x_on_0, x_on_1 = x.unwrapped

        with tf.device(prot.server_0.device_name):
            equal_zero_on_0 = x_on_0.equal_zero(out_dtype)

        with tf.device(prot.server_1.device_name):
            equal_zero_on_1 = x_on_1.equal_zero(out_dtype)

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
