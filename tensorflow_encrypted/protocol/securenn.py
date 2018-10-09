from __future__ import absolute_import
from typing import List, Optional
import random
import sys

import tensorflow as tf

from .protocol import memoize
from ..protocol.pond import (
    Pond, PondTensor, PondPublicTensor, PondPrivateTensor, PondMaskedTensor
)
from ..tensor import PrimeTensor, Int32Tensor
from ..tensor.prime import prime_factory as gen_prime_factory
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
        prime_factory: AbstractFactory = None,
        odd_factory: AbstractFactory = None,
        **kwargs
    ) -> None:
        super(SecureNN, self).__init__(
            server_0=server_0,
            server_1=server_1,
            crypto_producer=server_2,
            **kwargs
        )
        self.server_2 = self.crypto_producer
        self.prime_factory = prime_factory or gen_prime_factory(67)  # TODO: import or choose based on factory kwarg to super.__init__()
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
        if self.M % 2 != 1:
            # NOTE: this is currently only for use with an odd-modulus CRTTensor
            #       NativeTensor will use an even modulus and will require share_convert
            raise Exception('SecureNN protocol assumes a ring of odd cardinality, ' +
                            'but it was initialized with an even one.')
        return self.lsb(x * 2)

    @memoize
    def lsb(self, x: PondTensor) -> PondTensor:
        return self.dispatch('lsb', x, container=_thismodule)

    @memoize
    def bits(self, x: PondTensor, prime: Optional[int] = None) -> 'PondTensor':
        return self.dispatch('bits', x, container=_thismodule, prime=prime)

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

    def factory_from_type(self, type: str) -> AbstractFactory:
        if type == 'prime':
            return self.prime_factory

        return self.tensor_factory

    def _private_compare(self, x_bits: PondPrivateTensor, r: PondPublicTensor, beta: PondPublicTensor):
        base_shape = r.shape
        bit_length = x_bits.shape[-1]

        assert base_shape == x_bits.shape[:-1]
        assert base_shape == beta.shape

        with tf.name_scope('private_compare'):

            # use either r and t = r + 1 according to beta
            s = self.select(beta, r, r + 1)
            s_bits = s.to_bits(self.prime_factory.modulus)

            # compute w_sum
            w_bits = self.bitwise_xor(x_bits, s_bits)
            w_sum = self.cumsum(w_bits, axis=-1, reverse=True, exclusive=True)

            # compute c, ignoring edge cases at first
            sign = self.select(beta, 1, -1)
            sign = self.expand_dims(sign, axis=-1)
            c_except_edge_case = (s_bits - x_bits) * sign + 1 + w_sum

            # adjust for edge cases, i.e. where beta is 1 and s is zero (meaning r was -1)
            edge_cases = self.bitwise_and(
                beta,
                self.equal_zero(s)
            )
            edge_cases = self.expand_dims(edge_cases, axis=-1)
            c_edge_case_raw = tf.constant([0] + [1] * (bit_length-1), dtype=tf.int32, shape=(1, bit_length))
            c_edge_case = PondPrivateTensor(self, *self._share(PrimeTensor(c_edge_case_raw, 37), self.prime_factory), False)
            c = self.select(
                edge_cases,
                c_except_edge_case,
                c_edge_case
            )  # type: PondPrivateTensor

            # generate multiplicative mask to hide non-zero values
            with tf.device(self.server_0.device_name):
                mask_raw = self.prime_factory.Tensor.sample_uniform(c.shape, minval=1)
                mask = PondPublicTensor(self, mask_raw, mask_raw, False)

            # mask non-zero values; this is safe when we're in a field
            c_masked = c * mask

            # TODO[Morten] permute

            # reconstruct masked values on server 2 to find entries with zeros
            with tf.device(self.server_2.device_name):
                d = self._reconstruct(*c_masked.unwrapped)
                # find all zero entries
                zeros = d.equal_zero()
                # for each bit sequence, determine whether it has one or no zero in it
                rows_with_zeros = zeros.reduce_sum(axis=-1, keepdims=False)
                # reshare result
                return PondPrivateTensor(self, *self._share(rows_with_zeros, self.prime_factory), False)

    def equal_zero(self, x):
        # TODO[Morten] use devices, except to int100 etc
        value = x.value_on_0.equal_zero()
        return PondPublicTensor(self, value, value, False)

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


def _lsb_private(prot: SecureNN, y: PondPrivateTensor):

    with tf.name_scope('lsb'):

        with tf.name_scope('lsb_mask'):

            with tf.device(prot.server_2.device_name):
                x = prot.tensor_factory.Tensor.sample_uniform(y.shape)
                xbits = x.to_bits()

                xbits.value = tf.Print(xbits.value, [xbits.value], 'X BITS!', summarize=50)

                xlsb = xbits[..., 0]

                x = PondPrivateTensor(prot, *prot._share(x, factory=prot.odd_factory), is_scaled=False)
                xbits = PondPrivateTensor(prot, *prot._share(xbits, factory=prot.prime_factory), is_scaled=False)
                xlsb = PondPrivateTensor(prot, *prot._share(xlsb, factory=prot.tensor_factory), is_scaled=False)
                # xlsb.share0.value = tf.Print(xlsb.share0.value, [xlsb.reveal().value_on_0.value], 'xlsb', summarize=10)

            devices = [prot.server_0.device_name, prot.server_1.device_name]
            bits_device = random.choice(devices)
            with tf.device(bits_device):
                # TODO[Morten] pull this out as a separate `sample_bits` method on tensors (optimized for bits only)
                backing = prot.tensor_factory.Tensor.sample_bounded(y.shape, 1)
                beta = PondPublicTensor(prot, backing, backing, is_scaled=False)

        r = (y + x).reveal()
        r0, r1 = r.unwrapped

        # TODO[Morten] wrap this in a `to_bits()` on public tensors?
        with tf.device(prot.server_0.device_name):
            rbits0 = r0.to_bits()

        with tf.device(prot.server_1.device_name):
            rbits1 = r1.to_bits()

        rbits = PondPublicTensor(prot, rbits0, rbits1, is_scaled=False)
        rlsb = rbits[..., 0]

        print('backing tensor', xbits.share0)
        xbits.share0.value = tf.Print(xbits.share0.value, [xbits.reveal().value_on_1.value], 'X BITS before PC!', summarize=50)
        bp = prot._private_compare(xbits, r, beta)
        print('inputs', x.share0, r, beta)
        # bp = prot.private_compare(x, r, beta)
        # bp.share0.value = tf.Print(bp.share0.value, [bp.reveal().value_on_0.value], 'bpsh', summarize=10)

        gamma = prot.bitwise_xor(bp, beta)
        gamma.share0.value = tf.Print(gamma.share0.value, [gamma.reveal().value_on_0.value], 'gamm', summarize=10)
        delta = prot.bitwise_xor(xlsb, rlsb)
        delta.share0.value = tf.Print(delta.share0.value, [delta.reveal().value_on_0.value], 'delt', summarize=10)
        alpha = prot.bitwise_xor(gamma, delta)
        alpha.share0.value = tf.Print(alpha.share0.value, [alpha.reveal().value_on_0.value], 'alph', summarize=10)
        print(bp, beta, xlsb, rlsb)
        print(gamma.share0, delta.share0, alpha.share0)
        return alpha


def _lsb_masked(prot: SecureNN, x: PondMaskedTensor):
    return prot.lsb(x.unmasked)


def _bits_public(prot: SecureNN, x: PondPublicTensor, factory: Optional[AbstractFactory] = None) -> PondPublicTensor:

    factory = factory or prot.prime_factory

    with tf.name_scope('bits'):

        x_on_0, x_on_1 = x.unwrapped

        with tf.device(prot.server_0.device_name):
            bits_on_0 = x_on_0.to_bits(factory)

        with tf.device(prot.server_1.device_name):
            bits_on_1 = x_on_1.to_bits(factory)

        return PondPublicTensor(prot, bits_on_0, bits_on_1, x.is_scaled)
