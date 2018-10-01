from typing import List, Tuple, Optional
import random
import sys
import tensorflow as tf
import numpy as np

from .protocol import memoize
from ..protocol.pond import (
    Pond, PondTensor, PondPublicTensor, PondPrivateTensor, PondMaskedTensor
)
from ..tensor.shared import binarize
from ..tensor.prime import prime_factory, PrimeTensor
from ..tensor.factory import AbstractFactory
from ..player import Player
from ..tensor.tensor import AbstractTensor

_thismodule = sys.modules[__name__]
p = 67


class SecureNN(Pond):

    def __init__(
        self,
        server_0: Player,
        server_1: Player,
        server_2: Player,
        alt_factory: AbstractFactory=prime_factory(p),
        **kwargs
    ) -> None:
        super(SecureNN, self).__init__(
            server_0=server_0,
            server_1=server_1,
            crypto_producer=server_2,
            **kwargs
        )
        self.alt_factory = alt_factory

    @memoize
    def bitwise_not(self, x: PondTensor) -> PondTensor:
        assert not x.is_scaled, "Input is not supposed to be scaled"
        return self.sub(1, x)

    @memoize
    def bitwise_and(self, x: PondTensor, y: PondTensor) -> PondTensor:
        assert not x.is_scaled, "Input is not supposed to be scaled"
        assert not y.is_scaled, "Input is not supposed to be scaled"
        return x * y

    @memoize
    def bitwise_or(self, x: PondTensor, y: PondTensor) -> PondTensor:
        assert not x.is_scaled, "Input is not supposed to be scaled"
        assert not y.is_scaled, "Input is not supposed to be scaled"
        return x + y - self.bitwise_and(x, y)

    @memoize
    def bitwise_xor(self, x: PondTensor, y: PondTensor) -> PondTensor:
        assert not x.is_scaled, "Input is not supposed to be scaled"
        assert not y.is_scaled, "Input is not supposed to be scaled"
        return x + y - self.bitwise_and(x, y) * 2

    def binarize(self, x: PondPublicTensor, modulus: Optional[int]=None) -> PondTensor:
        modulus = modulus or p
        return binarize(x.value_on_0, modulus)

    @memoize
    def msb(self, x: PondTensor) -> PondTensor:
        # NOTE when the modulus is odd then msb reduces to lsb via x -> 2*x
        if self.M % 2 != 1:
            # NOTE: this is only for use with an odd-modulus CRTTensor
            #       NativeTensor will use an even modulus and will require share_convert
            raise Exception('SecureNN protocol assumes a ring of odd cardinality, ' +
                            'but it was initialized with an even one.')
        return self.lsb(x * 2)

    def lsb(self, x: PondTensor) -> PondTensor:
        return self.dispatch('lsb', x, container=_thismodule)

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
    def select_share(self, x: PondTensor, y: PondTensor, bit: PondTensor) -> PondTensor:
        return x + bit * (y - x)

    def private_compare(self, x: PondTensor, r: PondTensor, beta: PondTensor) -> PondTensor:
        raise NotImplementedError()

    def share_convert(self, x: PondPrivateTensor) -> PondPrivateTensor:
        L = self.tensor_factory.Tensor.modulus

        if L > 2**64:
            raise Exception('SecureNN share convert only support moduli of less or equal to 2 ** 64.')

        # P0
        with tf.device(self.server_0.device_name):
            bitmask = _generate_random_bits(self, [1])
            sharemask = self.tensor_factory.Tensor.sample_uniform(x.shape) + 1

            sharemask0, sharemask1, alpha_wrap = share_with_wrap(self, sharemask, L)

            pvt_sharemask = PondPrivateTensor(self, sharemask0, sharemask1, is_scaled=False)

            masked = x + pvt_sharemask

        alpha_wrap_t = PrimeTensor(-alpha_wrap.value - 1, p)

        # P0, P1
        with tf.device(self.server_0.device_name):
            beta_wrap_0 = x.share0.compute_wrap(sharemask0, L)

        with tf.device(self.server_1.device_name):
            beta_wrap_1 = x.share1.compute_wrap(sharemask1, L)

        beta_wrap = PondPrivateTensor(self, beta_wrap_0, beta_wrap_1, is_scaled=False)

        # P2
        with tf.device(self.crypto_producer.device_name):
            delta_wrap = masked.share0.compute_wrap(masked.share1, L)
            x_pub_masked = masked.reveal()

            #xbits = binarize(x_pub_masked.value_on_0)
            #share0, share1 = self._share(xbits, self.alt_factory)
            #bitshares = PondPrivateTensor(self, share0, share1, is_scaled=False)

            share0, share1 = self._share(delta_wrap)
            deltashares = PondPrivateTensor(self, share0, share1, is_scaled=False)

            # outbit = self.private_compare(bitshares, pvt_sharemask.reveal().value_on_0 - 1, bitmask)
            inp = self.tensor_factory.Tensor.from_native(np.array([1]))
            outbit = PondPublicTensor(self, inp, inp, is_scaled=False)

            compared0, compared1 = self._share(outbit.value_on_0)

        compared = PondPrivateTensor(self, compared0, compared1, is_scaled=False)

        # P0, P1
        preconverter = self.bitwise_xor(compared, bitmask)

        converter = deltashares + beta_wrap + preconverter

        converter.share0 += alpha_wrap_t

        return x - converter

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
            with tf.device(prot.crypto_producer.device_name):
                x = prot.tensor_factory.Tensor.sample_uniform(y.shape)
                xbits = binarize(x, p)
                xlsb = xbits[..., 0]
                x = PondPrivateTensor(prot, *prot._share(x, prot.tensor_factory), is_scaled=False)
                xbits = PondPrivateTensor(prot, *prot._share(xbits, prot.alt_factory), is_scaled=False)
                xlsb = PondPrivateTensor(prot, *prot._share(xlsb, prot.tensor_factory), is_scaled=False)

            devices = [prot.server_0.device_name, prot.server_1.device_name]
            bits_device = random.choice(devices)
            with tf.device(bits_device):
                b = _generate_random_bits(prot, y.shape)

        with tf.name_scope('lsb_ops'):
            r = (y + x).reveal()
            rbits = prot.binarize(r, p)
            rlsb = rbits[..., 0]

            bp = prot.private_compare(xbits, r, b)

            gamma = prot.bitwise_xor(bp, b)
            delta = prot.bitwise_xor(xlsb, rlsb)

            alpha = prot.bitwise_xor(gamma, delta)

            return alpha


def _lsb_masked(prot: SecureNN, x: PondMaskedTensor):
    return prot.lsb(x.unmasked)


def _generate_random_bits(prot: SecureNN, shape: List[int]):
    backing = prime_factory(2).Tensor.sample_uniform(shape)
    return PondPublicTensor(prot, backing, backing, is_scaled=False)


def share_with_wrap(prot: SecureNN, sample: PrimeTensor,
                    modulus: int) -> Tuple[AbstractTensor, AbstractTensor, AbstractTensor]:
    x, y = prot._share(sample)
    kappa = x.compute_wrap(y, modulus)
    return x, y, kappa
