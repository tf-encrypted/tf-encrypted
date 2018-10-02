from __future__ import absolute_import
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
from ..config import get_default_config
from tensorflow_encrypted.tensor.int32 import Int32Factory, Int32Tensor
import numpy as np
bits = 32

_thismodule = sys.modules[__name__]
p = 67  # TODO: import or choose based on factory kwarg to super.__init__()


class SecureNN(Pond):

    def __init__(
        self,
        server_0: Player,
        server_1: Player,
        server_2: Player,
        alt_factory: AbstractFactory = prime_factory(p),
        **kwargs
    ) -> None:
        super(SecureNN, self).__init__(
            server_0=server_0 or get_default_config().get_player('server0'),
            server_1=server_1 or get_default_config().get_player('server1'),
            crypto_producer=server_2 or get_default_config().get_player('crypto_producer'),
            **kwargs
        )
        self.alt_factory = alt_factory

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

    def _private_compare_beta0(self, zeros: PondPublicTensor, input: PondPrivateTensor, rho: PondPublicTensor):

        # TODO -- gather not working (runtime error)
        #         not a probelm right now as the test is only using b == 0
        # input = self.gather(input, zeros)
        # rho = self.gather(rho, zeros)

        w = self.bitwise_xor(input, rho)

        with tf.device(self.server_0.device_name):
            w0_sum = tf.zeros(shape=w.shape, dtype=tf.int32)
            for i in range(bits - 1, -1, -1):
                sum = self.sum(w[:, i+1:], axis=1)
                indices = []

                for j in range(0, w.shape.as_list()[0]):
                    indices.append([j, i])

                update_0 = tf.SparseTensor(indices, sum.share0.value, w.shape)

            w0_sum = w0_sum + tf.sparse_tensor_to_dense(update_0)

        with tf.device(self.server_1.device_name):
            w1_sum = tf.zeros(shape=w.shape, dtype=tf.int32)
            for i in range(bits - 1, -1, -1):
                sum = self.sum(w[:, i+1:], axis=1)
                indices = []

                for j in range(0, w.shape.as_list()[0]):
                    indices.append([j, i])

                update_1 = tf.SparseTensor(indices, sum.share1.value, w.shape)

            w1_sum = w1_sum + tf.sparse_tensor_to_dense(update_1)

        w_sum = PondPrivateTensor(self, Int32Tensor(w0_sum), Int32Tensor(w1_sum), w.is_scaled)

        c = rho - input + 1 + w_sum
        return c

    def _private_compare_beta1(self, ones: PondPublicTensor, input: PondPrivateTensor, theta: PondPublicTensor):

        w = self.bitwise_xor(input, theta)

        with tf.device(self.server_0.device_name):
            w0_sum = tf.zeros(shape=w.shape, dtype=tf.int32)
            for i in range(bits - 1, -1, -1):
                sum = self.sum(w[:, i+1:], axis=1)
                indices = []

                for j in range(0, w.shape.as_list()[0]):
                    indices.append([j, i])

                update_0 = tf.SparseTensor(indices, sum.share0.value, w.shape)

            w0_sum = w0_sum + tf.sparse_tensor_to_dense(update_0)

        with tf.device(self.server_1.device_name):
            w1_sum = tf.zeros(shape=w.shape, dtype=tf.int32)
            for i in range(bits - 1, -1, -1):
                sum = self.sum(w[:, i+1:], axis=1)
                indices = []

                for j in range(0, w.shape.as_list()[0]):
                    indices.append([j, i])

                update_1 = tf.SparseTensor(indices, sum.share1.value, w.shape)

            w1_sum = w1_sum + tf.sparse_tensor_to_dense(update_1)

        w_sum = PondPrivateTensor(self, Int32Tensor(w0_sum), Int32Tensor(w1_sum), w.is_scaled)

        c = input - theta + 1 + w_sum
        return c

    def _private_compare_edge(self):
        random_values_u = self.tensor_factory.Tensor.sample_random_tensor((bits,), modulus=p - 1) + 1
        c0 = (random_values_u + 1)
        c1 = (-random_values_u)

        c0[0] = random_values_u[0]

        return c0

    def private_compare(self, input: PondPrivateTensor, rho: PondPublicTensor, beta: PondPublicTensor):
        with tf.name_scope('private_compare'):
            theta = (rho + 1)

            beta = beta.reshape([4, 1])
            beta = beta.broadcast([4, 32])

            with tf.name_scope('find_zeros'):
                eq = self.equal(beta, 0)
                zeros = self.where(eq)

            with tf.name_scope('find_ones'):
                eq = self.equal(beta, 1)
                ones = self.where(eq)

            with tf.name_scope('find_edges'):
                edges = self.where(self.equal(rho, 2 ** bits - 1))

            # return tf.Print(zeros.value_on_1.value, [zeros.value_on_1.value], 'ONES:', summarize=1000)

            # with tf.name_scope('find_non_edge_ones'):
            #     ones = tf.setdiff1d(ones, edges)

            pc_0 = self._private_compare_beta0(zeros, input, rho)
            pc_1 = self._private_compare_beta1(ones, input, theta)

            # c0_edge, c1_edge = self._private_compare_edge()

            pc_0 = pc_0.reshape([-1])
            pc_1 = pc_1.reshape([-1])

            # convert ones/zeros/edges into (4,32) of the same values so it will
            # work with sparse tensor (e.g.)
            #   [1, 0, 1, 0]
            # will turn into
            #   [[1,1,..,1], [0,0,..,0], [1,1,..,1], [0,0,..,0]]
            # z_long = tf.ones(shape=(4, 32), dtype=tf.int32)
            # zeros = beta.reshape([4, 1]) * PondPublicTensor(self, value_on_0=Int32Tensor(z_long), value_on_1=Int32Tensor(z_long), is_scaled=beta.is_scaled)
            #
            # o_long = tf.ones(shape=(4, 32), dtype=tf.int32)
            # ones = beta.reshape([4, 1]) * PondPublicTensor(self, value_on_0=Int32Tensor(o_long), value_on_1=Int32Tensor(o_long), is_scaled=beta.is_scaled)


            # return tf.Print(ones.value_on_1.value, [ones.value_on_1.value], 'ZEROS:', summarize=50)

            with tf.device(self.server_0.device_name):
                c0 = tf.zeros(shape=input.shape, dtype=tf.int32)

                # return tf.Print(pc_1.share0.value, [pc_1.share0.value], 'zeros:', summarize=50)

                delta0 = tf.SparseTensor(zeros.value_on_0.value, pc_0.share0.value, input.shape)
                # delta1 = tf.SparseTensor(ones.value_on_0.value, pc_1.share0.value, input.shape)

                c0 = c0 + tf.sparse_tensor_to_dense(delta0)  # + tf.sparse_tensor_to_dense(delta1)
                c0 = Int32Tensor(c0)

            with tf.device(self.server_0.device_name):
                c1 = tf.zeros(shape=input.shape, dtype=tf.int32)

                delta0 = tf.SparseTensor(zeros.value_on_0.value, pc_0.share1.value, input.shape)
                # delta1 = tf.SparseTensor(ones.value_on_1.value, pc_1.share1.value, input.shape)

                c1 = c1 + tf.sparse_tensor_to_dense(delta0)  # + tf.sparse_tensor_to_dense(delta1)
                c1 = Int32Tensor(c1)

            answer = PondPrivateTensor(self, share0=c0, share1=c1, is_scaled=input.is_scaled)
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

        bp = prot.private_compare(xbits, r, b)

        gamma = prot.bitwise_xor(bp, b)
        delta = prot.bitwise_xor(xlsb, rlsb)

        alpha = prot.bitwise_xor(gamma, delta)

        return alpha


def _lsb_masked(prot: SecureNN, x: PondMaskedTensor):
    return prot.lsb(x.unmasked)


def _generate_random_bits(prot: SecureNN, shape: List[int]):
    backing = prot.tensor_factory.Tensor.sample_uniform(shape)
    return PondPublicTensor(prot, backing, backing, is_scaled=False)
