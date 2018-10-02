from typing import List
import random
import sys
import tensorflow as tf
from .protocol import memoize
from ..protocol.pond import (
    Pond, PondTensor, PondPublicTensor, PondPrivateTensor, PondMaskedTensor
)
from ..tensor.prime import prime_factory
from ..tensor.factory import AbstractFactory
from ..player import Player

_thismodule = sys.modules[__name__]
p = 37  # TODO: import or choose based on factory kwarg to super.__init__()


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

    @memoize
    def msb(self, x: PondTensor) -> PondTensor:
        # NOTE when the modulus is odd then msb reduces to lsb via x -> 2*x
        if self.M % 2 != 1:
            # NOTE: this is currently only for use with an odd-modulus CRTTensor
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

    def private_compare(self,
                        x: PondPrivateTensor,
                        r: PondPublicTensor,
                        beta: PondPublicTensor) -> PondPrivateTensor:
        # this is a placeholder;
        # it computes the functionality of private_compare in plain text
        x = x.reveal()
        xval = self._reconstruct(x.value_on_0, x.value_on_1)
        rval = self._reconstruct(r.value_on_0, r.value_on_1)
        bval = tf.cast(self._reconstruct(beta.value_on_0, beta.value_on_1).value, tf.int8)
        tf_res = tf.cast(xval.value > rval.value, tf.int8)
        xord = tf.bitwise.bitwise_xor(tf_res, bval)
        val = self.tensor_factory.Tensor.from_native(tf.cast(xord, tf.int32))
        share0, share1 = self._share(val)
        return PondPrivateTensor(self, share0, share1, is_scaled=x.is_scaled)

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
            with tf.device(prot.crypto_producer.device_name):
                x = prot.tensor_factory.Tensor.sample_uniform(y.shape)
                xbits = x.to_bits()
                xlsb = xbits[..., 0]
                x = PondPrivateTensor(prot, *prot._share(x, prot.tensor_factory), is_scaled=False)
                xbits = PondPrivateTensor(prot, *prot._share(xbits, prot.alt_factory), is_scaled=False)
                xlsb = PondPrivateTensor(prot, *prot._share(xlsb, prot.tensor_factory), is_scaled=False)

            devices = [prot.server_0.device_name, prot.server_1.device_name]
            bits_device = random.choice(devices)
            with tf.device(bits_device):
                b = _generate_random_bits(prot, y.shape)

        r = (y + x).reveal()
        r0, r1 = r.unwrapped
        rbits0, rbits1 = r0.to_bits(), r1.to_bits()
        rbits = PondPublicTensor(prot, rbits0, rbits1, is_scaled=False)
        rlsb = rbits[..., 0]

        # bp = prot.private_compare(xbits, r, b)
        bp = prot.private_compare(x, r, b)

        gamma = prot.bitwise_xor(bp, b)
        delta = prot.bitwise_xor(xlsb, rlsb)

        alpha = prot.bitwise_xor(gamma, delta)

        return alpha


def _lsb_masked(prot: SecureNN, x: PondMaskedTensor):
    return prot.lsb(x.unmasked)


def _generate_random_bits(prot: SecureNN, shape: List[int]):
    backing = prot.tensor_factory.Tensor.sample_uniform(shape)
    return PondPublicTensor(prot, backing, backing, is_scaled=False)
