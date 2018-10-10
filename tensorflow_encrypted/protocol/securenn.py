from __future__ import absolute_import
from typing import Optional
import random
import sys

import tensorflow as tf

from .protocol import memoize
from ..protocol.pond import (
    Pond, PondTensor, PondPublicTensor, PondPrivateTensor, PondMaskedTensor
)
from ..tensor.prime import PrimeFactory
from ..tensor.factory import AbstractFactory
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
        return x + y - self.bitwise_and(x, y) * 2

    @memoize
    def msb(self, x: PondTensor) -> PondTensor:
        # NOTE when the modulus is odd then msb reduces to lsb via x -> 2*x
        if x.modulus % 2 != 1:
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
        # NOTE MSB is 1 iff xi < 0
        return self.msb(x)

    @memoize
    def non_negative(self, x: PondTensor) -> PondTensor:
        return self.bitwise_not(self.msb(x))

    @memoize
    def less(self, x: PondTensor, y: PondTensor) -> PondTensor:
        return self.negative(x - y)

    @memoize
    def less_equal(self, x: PondTensor, y: PondTensor) -> PondTensor:
        return self.bitwise_not(self.greater(x, y))

    @memoize
    def greater(self, x: PondTensor, y: PondTensor) -> PondTensor:
        return self.negative(y - x)

    @memoize
    def greater_equal(self, x: PondTensor, y: PondTensor) -> PondTensor:
        return self.bitwise_not(self.less(x, y))

    @memoize
    def select(self, choice_bit: PondTensor, x: PondTensor, y: PondTensor) -> PondTensor:
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
        drelu = self.non_negative(x)
        return drelu * x

    def max_pool(self, x):
        raise NotImplementedError

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
                x = y.backing_dtype.sample_uniform(y.shape)
                xbits = x.to_bits(factory=prot.prime_factory)
                xlsb = xbits[..., 0].cast(y.backing_dtype)

                x = PondPrivateTensor(prot, *prot._share(x), is_scaled=False)
                xbits = PondPrivateTensor(prot, *prot._share(xbits), is_scaled=False)
                xlsb = PondPrivateTensor(prot, *prot._share(xlsb), is_scaled=False)
                # xlsb.share0.value = tf.Print(xlsb.share0.value, [xlsb.reveal().value_on_0.value], 'xlsb', summarize=10)

            devices = [prot.server_0.device_name, prot.server_1.device_name]
            bits_device = random.choice(devices)
            with tf.device(bits_device):
                # TODO[Morten] pull this out as a separate `sample_bits` method on tensors (optimized for bits only)
                beta_raw = prot.prime_factory.sample_bounded(y.shape, 1)
                beta = PondPublicTensor(prot, beta_raw, beta_raw, is_scaled=False)

        r = (y + x).reveal()
        r0, r1 = r.unwrapped

        # TODO[Morten] wrap this in a `to_bits()` on public tensors?
        with tf.device(prot.server_0.device_name):
            rbits0 = r0.to_bits()

        with tf.device(prot.server_1.device_name):
            rbits1 = r1.to_bits()

        rbits = PondPublicTensor(prot, rbits0, rbits1, is_scaled=False)
        rlsb = rbits[..., 0]

        bp = _private_compare(prot, xbits, r, beta)
        # bp = prot.private_compare(x, r, beta)
        # bp.share0.value = tf.Print(bp.share0.value, [bp.reveal().value_on_0.value], 'bpsh', summarize=10)

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

        # use either r and t = r + 1 according to beta
        s = prot.select(beta.cast_backing(r.backing_dtype), r, r + 1)
        s_bits = prot.bits(s, factory=prime_dtype)
        assert s_bits.shape[-1] == bit_length

        # compute w_sum
        w_bits = prot.bitwise_xor(x_bits, s_bits)
        w_sum = prot.cumsum(w_bits, axis=-1, reverse=True, exclusive=True)
        assert w_sum.backing_dtype == prime_dtype

        # compute c, ignoring edge cases at first)
        sign = prot.select(beta, 1, -1)
        sign = prot.expand_dims(sign, axis=-1)
        c_except_edge_case = (s_bits - x_bits) * sign + 1 + w_sum
        assert c_except_edge_case.backing_dtype == prime_dtype

        # adjust for edge cases, i.e. where beta is 1 and s is zero (meaning r was -1)
        edge_cases = prot.bitwise_and(
            beta,
            prot.equal_zero(s, prime_dtype)
        )
        edge_cases = prot.expand_dims(edge_cases, axis=-1)
        c_edge_case_raw = prime_dtype.tensor(tf.constant([0] + [1] * (bit_length - 1), dtype=tf.int32, shape=(1, bit_length)))
        c_edge_case = PondPrivateTensor(prot, *prot._share(c_edge_case_raw), False)
        c = prot.select(
            edge_cases,
            c_except_edge_case,
            c_edge_case
        )  # type: PondPrivateTensor
        assert c.backing_dtype == prime_dtype

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
            result = PondPrivateTensor(prot, *prot._share(rows_with_zeros), False)

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
