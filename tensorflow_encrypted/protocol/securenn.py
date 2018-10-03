from typing import List, Tuple
import random
import sys
import tensorflow as tf
import numpy as np

from .protocol import memoize
from ..protocol.pond import (
    Pond, PondTensor, PondPublicTensor, PondPrivateTensor, PondMaskedTensor
)
from ..tensor.prime import prime_factory as gen_prime_factory
from ..tensor.factory import AbstractFactory
from ..player import Player
from ..tensor.tensor import AbstractTensor
from ..config import get_default_config
from ..tensor.odd_implicit import OddImplicitTensor
from ..tensor.int32 import Int32Tensor

_thismodule = sys.modules[__name__]


class SecureNN(Pond):

    def __init__(
        self,
        server_0: Player,
        server_1: Player,
        server_2: Player,
        prime_factory: AbstractFactory=None,
        odd_factory: AbstractFactory=None,
        **kwargs
    ) -> None:
        super(SecureNN, self).__init__(
            server_0=server_0 or get_default_config().get_player('server0'),
            server_1=server_1 or get_default_config().get_player('server1'),
            crypto_producer=server_2 or get_default_config().get_player('crypto_producer'),
            **kwargs
        )
        self.server_2 = server_2
        self.prime_factory = prime_factory or gen_prime_factory(37)  # TODO: import or choose based on factory kwarg to super.__init__()
        self.odd_factory = odd_factory or self.tensor_factory

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

    def odd_modulus_bitwise_xor(self, x: PondPrivateTensor, bits: PondPublicTensor) -> PondPrivateTensor:
        int_type = self.tensor_factory.Tensor.int_type

        with tf.device(self.server_0.device_name):
            ones = OddImplicitTensor(tf.ones(x.shape, dtype=int_type), dtype=int_type)
            share0 = ones.optional_sub(x.share0, bits.value_on_0)

        with tf.device(self.server_1.device_name):
            zeros = OddImplicitTensor(tf.zeros(x.shape, dtype=int_type), dtype=int_type)
            share1 = zeros.optional_sub(x.share1, bits.value_on_1)

        return PondPrivateTensor(self, share0, share1, is_scaled=False)

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
        xval = x.value_on_0
        rval = r.value_on_1
        bval = tf.cast(beta.value_on_0.value, tf.int8)
        tf_res = tf.cast(xval.value > rval.value, tf.int8)
        tf_res = tf.Print(tf_res, [xval.value, rval.value], 'x and r', summarize=10)
        tf_res = tf.Print(tf_res, [tf_res], 'resu', summarize=10)
        tf_res = tf.Print(tf_res, [bval], 'bval', summarize=10)
        xord = tf.bitwise.bitwise_xor(tf_res, bval)
        xord = tf.Print(xord, [xord], 'xord', summarize=10)
        val = self.tensor_factory.Tensor.from_native(tf.cast(xord, tf.int32))

        share0, share1 = self._share(val)
        shared = PondPrivateTensor(self, share0, share1, is_scaled=x.is_scaled)
        shared.share0.value = tf.Print(shared.share0.value, [shared.reveal().value_on_0.value], 'shared res', summarize=10)
        return shared

    def share_convert(self, x: PondPrivateTensor) -> PondPrivateTensor:
        L = self.tensor_factory.Tensor.modulus

        if L > 2**64:
            raise Exception('SecureNN share convert only support moduli of less or equal to 2 ** 64.')

        # P0
        with tf.device(self.server_0.device_name):
            bitmask = _generate_random_bits(self, x.shape)
            sharemask = self.tensor_factory.Tensor.sample_uniform(x.shape)

            sharemask0, sharemask1, alpha_wrap = share_with_wrap(self, sharemask, L)

            pvt_sharemask = PondPrivateTensor(self, sharemask0, sharemask1, is_scaled=False)

            masked = x + pvt_sharemask

        alpha_wrap_t = PrimeTensor(-alpha_wrap.value - 1, p)
        zero = PrimeTensor(np.zeros(alpha_wrap.shape, dtype=np.int32), p)
        alpha = PondPublicTensor(self, alpha_wrap_t, zero, is_scaled=False)

        # P0, P1
        with tf.device(self.server_0.device_name):
            beta_wrap_0 = x.share0.compute_wrap(sharemask0, L)

        with tf.device(self.server_1.device_name):
            beta_wrap_1 = x.share1.compute_wrap(sharemask1, L)

        beta_wrap = PondPublicTensor(self, beta_wrap_0, beta_wrap_1, is_scaled=False)

        # P2
        with tf.device(self.crypto_producer.device_name):
            delta_wrap = masked.share0.compute_wrap(masked.share1, L)
            x_pub_masked = masked.reveal()

            xbits = x_pub_masked.value_on_0  # .to_bits()

        deltashare0, deltashare1 = self._share(delta_wrap)
        bitshare0, bitshare1 = self._share(xbits)

        bitshares = PondPrivateTensor(self, bitshare0, bitshare1, is_scaled=False)
        deltashares = PondPrivateTensor(self, deltashare0, deltashare1, is_scaled=False)

        with tf.device(self.server_0.device_name):
            compared = self.private_compare(bitshares, pvt_sharemask.reveal() - 1, bitmask)

        with tf.device(self.server_1.device_name):
            compared.share1.value = tf.identity(compared.share1.value)

        # P0, P1
        print(bitmask.value_on_0.value)
        print(alpha.value_on_0.value)
        print(beta_wrap.value_on_0.value)
        preconverter = self.odd_modulus_bitwise_xor(compared, bitmask)

        deltashares = self.to_odd_modulus(deltashares)
        beta_wrap = self.to_odd_modulus(beta_wrap)

        # print(deltashares.share0.value, deltashares.share1.value)
        # print(preconverter.share0.value, preconverter.share1.value)
        #
        # one = deltashares.share0.value + preconverter.share0.value
        # two = deltashares.share1.value + preconverter.share1.value
        #
        # print(one)
        # print(two)

        # print(beta_wrap.value_on_0.value)
        # print(alpha.value_on_0.value)

        converter = preconverter + deltashares + beta_wrap

        converter = converter + self.to_odd_modulus(alpha)

        # print(x.share0.value)
        # print(x.share1.value)

        return self.to_odd_modulus(x) - converter

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

    def to_odd_modulus(self, x: PondTensor):
        if isinstance(x, PondPrivateTensor):
            with tf.device(self.server_0.device_name):
                share0 = x.share0.to_odd_modulus()

            with tf.device(self.server_1.device_name):
                share1 = x.share1.to_odd_modulus()

            return PondPrivateTensor(self, share0, share1, is_scaled=False)
        elif isinstance(x, PondPublicTensor):
            with tf.device(self.server_0.device_name):
                share0 = x.value_on_0.to_odd_modulus()

            with tf.device(self.server_1.device_name):
                share1 = x.value_on_1.to_odd_modulus()

            return PondPublicTensor(self, share0, share1, is_scaled=False)


def _lsb_private(prot: SecureNN, y: PondPrivateTensor):

    with tf.name_scope('lsb'):

        with tf.name_scope('lsb_mask'):

            with tf.device(prot.server_2.device_name):
                x = prot.tensor_factory.Tensor.sample_uniform(y.shape)
                xbits = x.to_bits()
                xlsb = xbits[..., 0]

                x = PondPrivateTensor(prot, *prot._share(x, factory=prot.odd_factory), is_scaled=False)
                xbits = PondPrivateTensor(prot, *prot._share(xbits, factory=prot.prime_factory), is_scaled=False)
                xlsb = PondPrivateTensor(prot, *prot._share(xlsb, factory=prot.tensor_factory), is_scaled=False)

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
        # bp = prot.private_compare(xbits, r, beta)
        bp = prot.private_compare(x, r, beta)

        gamma = prot.bitwise_xor(bp, beta)
        delta = prot.bitwise_xor(xlsb, rlsb)
        alpha = prot.bitwise_xor(gamma, delta)
        return alpha


def _lsb_masked(prot: SecureNN, x: PondMaskedTensor):
    return prot.lsb(x.unmasked)


def _generate_random_bits(prot: SecureNN, shape: List[int]):
    backing = gen_prime_factory(2).Tensor.sample_uniform(shape)
    return PondPublicTensor(prot, backing, backing, is_scaled=False)


def share_with_wrap(prot: SecureNN, sample: AbstractTensor,
                    modulus: int) -> Tuple[AbstractTensor, AbstractTensor, AbstractTensor]:
    x, y = prot._share(sample)
    kappa = x.compute_wrap(y, modulus)
    return x, y, kappa
