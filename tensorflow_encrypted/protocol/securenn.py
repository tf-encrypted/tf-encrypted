from __future__ import absolute_import
from typing import List
import random
import sys
import tensorflow as tf
from .protocol import memoize
from ..protocol.pond import (
    Pond, PondTensor, PondPublicTensor, PondPrivateTensor, PondMaskedTensor
)
from ..tensor.prime import prime_factory as gen_prime_factory
from ..tensor.factory import AbstractFactory
from ..player import Player
from ..config import get_default_config
from tensorflow_encrypted.tensor.int32 import Int32Tensor
bits = 32

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
    def select_share(self, x: PondTensor, y: PondTensor, choice_bit: PondTensor) -> PondTensor:
        return x + choice_bit * (y - x)

    def factory_from_type(self, type: str) -> AbstractFactory:
        if type == 'prime':
            return self.prime_factory

        return self.tensor_factory

    def _private_compare_beta0(self, input: PondPrivateTensor, rho: PondPublicTensor):

        w = self.bitwise_xor(input, rho)

        with tf.device(self.server_0.device_name):
            w0_sum = tf.zeros(shape=w.shape, dtype=tf.int32)
            for i in range(bits - 1, -1, -1):
                sum = self.sum(w[:, i + 1:], axis=1)
                indices = []

                for j in range(0, w.shape.as_list()[0]):
                    indices.append([j, i])

                update_0 = tf.SparseTensor(indices, sum.share0.value, w.shape)

            w0_sum = w0_sum + tf.sparse_tensor_to_dense(update_0)

        with tf.device(self.server_1.device_name):
            w1_sum = tf.zeros(shape=w.shape, dtype=tf.int32)
            for i in range(bits - 1, -1, -1):
                sum = self.sum(w[:, i + 1:], axis=1)
                indices = []

                for j in range(0, w.shape.as_list()[0]):
                    indices.append([j, i])

                update_1 = tf.SparseTensor(indices, sum.share1.value, w.shape)

            w1_sum = w1_sum + tf.sparse_tensor_to_dense(update_1)

        w_sum = PondPrivateTensor(self, self.prime_factory.Tensor.from_native(w0_sum), self.prime_factory.Tensor.from_native(w1_sum), w.is_scaled)

        c = rho - input + 1 + w_sum
        return c

    def _private_compare_beta1(self, input: PondPrivateTensor, theta: PondPublicTensor):

        w = self.bitwise_xor(input, theta)

        with tf.device(self.server_0.device_name):
            w0_sum = tf.zeros(shape=w.shape, dtype=tf.int32)
            for i in range(bits - 1, -1, -1):
                sum = self.sum(w[:, i + 1:], axis=1)
                indices = []

                for j in range(0, w.shape.as_list()[0]):
                    indices.append([j, i])

                update_0 = tf.SparseTensor(indices, sum.share0.value, w.shape)

            w0_sum = w0_sum + tf.sparse_tensor_to_dense(update_0)

        with tf.device(self.server_1.device_name):
            w1_sum = tf.zeros(shape=w.shape, dtype=tf.int32)
            for i in range(bits - 1, -1, -1):
                sum = self.sum(w[:, i + 1:], axis=1)
                indices = []

                for j in range(0, w.shape.as_list()[0]):
                    indices.append([j, i])

                update_1 = tf.SparseTensor(indices, sum.share1.value, w.shape)

            w1_sum = w1_sum + tf.sparse_tensor_to_dense(update_1)

        w_sum = PondPrivateTensor(self, self.prime_factory.Tensor.from_native(w0_sum), self.prime_factory.Tensor.from_native(w1_sum), w.is_scaled)

        c = input - theta + 1 + w_sum
        return c

    def _private_compare_edge(self):
        random_values_u = self.tensor_factory.Tensor.sample_random_tensor((bits,), modulus=p - 1) + 1
        c0 = (random_values_u + 1)
        c1 = (-random_values_u)

        c0[0] = random_values_u[0]

        return c0

    def private_compare(self, input: PondPrivateTensor, rho: PondPublicTensor, beta: PondPublicTensor):
        theta = rho + 1

        rho = rho.to_bits()
        theta = theta.to_bits()

        with tf.name_scope('private_compare'):
            beta = beta.reshape([beta.shape.as_list()[0], 1])
            beta = beta.broadcast([beta.shape.as_list()[0], 32])

            with tf.name_scope('find_zeros'):
                eq = self.equal(beta, 0)
                zeros = self.where(eq)

            with tf.name_scope('find_ones'):
                eq = self.equal(beta, 1)
                ones = self.where(eq)

            with tf.name_scope('find_edges'):
                edges = self.where(self.equal(rho, 2 ** bits - 1))

            # with tf.name_scope('find_non_edge_ones'):
            #     ones = tf.setdiff1d(ones, edges)

            # return input
            pc_0 = self._private_compare_beta0(input, rho)
            pc_1 = self._private_compare_beta1(input, theta)

            # return tf.Print(zeros.value_on_1.value, [zeros.value_on_1.value], 'ZEROS')

            pc_0 = self.gather_nd(pc_0, zeros)
            pc_1 = self.gather_nd(pc_1, ones)

            # c0_edge, c1_edge = self._private_compare_edge()

            pc_0 = pc_0.reshape([-1])
            pc_1 = pc_1.reshape([-1])

            with tf.device(self.server_0.device_name):
                c0 = tf.zeros(shape=input.shape, dtype=tf.int32)

                delta0 = tf.SparseTensor(zeros.value_on_0.value, pc_0.share0.value, input.shape)
                delta1 = tf.SparseTensor(ones.value_on_0.value, pc_1.share0.value, input.shape)

                c0 = c0 + tf.sparse_tensor_to_dense(delta0) + tf.sparse_tensor_to_dense(delta1)
                c0 = self.prime_factory.Tensor.from_native(c0)

            with tf.device(self.server_0.device_name):
                c1 = tf.zeros(shape=input.shape, dtype=tf.int32)

                delta0 = tf.SparseTensor(zeros.value_on_1.value, pc_0.share1.value, input.shape)
                delta1 = tf.SparseTensor(ones.value_on_1.value, pc_1.share1.value, input.shape)

                c1 = c1 + tf.sparse_tensor_to_dense(delta0) + tf.sparse_tensor_to_dense(delta1)
                c1 = self.prime_factory.Tensor.from_native(c1)

            with tf.device(self.crypto_producer.device_name):
                answer = PondPrivateTensor(self, share0=c0, share1=c1, is_scaled=input.is_scaled).reveal()
                reduced = tf.reduce_max(tf.cast(tf.equal(answer.value_on_0.value, 0), tf.int32), axis=-1)

                answer = PondPublicTensor(self, Int32Tensor(reduced), Int32Tensor(reduced), is_scaled=answer.is_scaled)

            return answer

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
        # bp = prot.private_compare(xbits, r, beta)
        bp = prot.private_compare(x, r, beta)
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
