from typing import List
import random
import sys
import tensorflow as tf
from .protocol import memoize
from ..protocol.pond import (
    Pond, PondTensor, PondPrivateTensor, PondMaskedTensor
)
from ..player import Player


_thismodule = sys.modules[__name__]

class SecureNN(Pond):

    def __init__(
        self,
        server_0: Player,
        server_1: Player,
        server_2: Player,
        **kwargs
    ) -> None:
        super(SecureNN, self).__init__(
            server_0=server_0,
            server_1=server_1,
            crypto_producer=server_2,
            **kwargs
        )

        if self.M % 2 != 1:
            # NOTE: this is only for use with an odd-modulus CRTTensor
            #       NativeTensor will use an even modulus and will require share_convert
            raise Exception('SecureNN protocol assumes a ring of odd cardinality, ' +
                            'but it was initialized with an even one.')

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

    def binarize(self, x: PondTensor) -> PondTensor:
        raise NotImplementedError

    @memoize
    def msb(self, x: PondTensor) -> PondTensor:
        # NOTE when the modulus is odd then msb reduces to lsb via x -> 2*x
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
        w = y - x
        c = bit * w

        return x + c + PondPrivateTensor.zero(x.prot, x.shape)

    def private_compare(self, x: PondTensor, r: PondTensor, beta: PondTensor) -> PondTensor:
        raise NotImplementedError

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
    with tf.name_scope('lsb_mask'):
        with tf.device(prot.crypto_producer.device_name):
            x = prot.tensor_factory.Tensor.sample_uniform(y.shape)
            xbits = x.binarize()
            xlsb = xbits[..., 0]
            xlsb0, xlsb1 = prot.share(xlsb, p)
            x = PondPrivateTensor(prot, *prot.share(x), is_scaled=True)
            xbits = PondPrivateTensor(prot, *prot.share(xbits), is_scaled=False)
            # TODO: Generate zero mask?

        devices = [prot.server0.device_name, prot.server1.device_name]
        bits_device = random.choice(devices)
        with tf.device(bits_device):
            b = _generate_random_bits(y.shape)
            b0, b1 = beta.unwrapped

    with tf.name_scope('lsb'):
        r = (y + x).reveal()
        rbits0, rbits1 = prot.binarize(r).unwrapped
        rlsb0, rlsb1 = rbits0[..., 0], rbits1[..., 0]


        bp = prot.private_compare(xbits, r, b)
        bp0, bp1 = bp.unwrapped

        with tf.device(prot.server0.device_name):
            gamma_on_0 = bp0 + b0 * bp0 * (-2)
            delta_on_0 = xlsb0 + rlsb0 * xlsb0 * (-2)

        with tf.device(prot.server1.device_name):
            gamma_on_1 = bp1 + b1 * (bp1 * (-2) + 1)
            delta_on_1 = xlsb1 + rlsb1 * (xlsb1 * (-2) + 1)

        gamma = PondPrivateTensor(prot, gamma_on_0, gamma_on_1, is_scaled=False)
        delta = PondPrivateTensor(prot, delta_on_0, delta_on_1, is_scaled=False)
        theta = gamma * delta

        alpha = gamma + delta + theta * (-2)  # TODO: add zero mask? # TODO: __rmul__

        return alpha


def _lsb_masked(prot: SecureNN, x: PondMaskedTensor):
    return prot.lsb(x.unmasked)


def _generate_random_bits(prot: SecureNN, shape: List[int]):
    backing = prot.tensor_factory.Tensor.sample_bounded(y.shape)
    return PondPublicTensor(prot, backing, backing, is_scaled=False)  # FIXME: better way to generate bits
