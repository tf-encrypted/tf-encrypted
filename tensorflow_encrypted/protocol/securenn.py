from __future__ import absolute_import

import tensorflow as tf
from typing import Optional

from .protocol import memoize
from ..protocol.pond import (
    Pond, PondTensor, PondPrivateTensor, PondPublicTensor
)
from ..player import Player
from ..config import get_default_config
from tensorflow_encrypted.tensor.int32 import Int32Factory, Int32Tensor
import numpy as np
bits = 32


class SecureNN(Pond):

    def __init__(
        self,
        server_0: Optional[Player] = None,
        server_1: Optional[Player] = None,
        server_2: Optional[Player] = None,
        **kwargs
    ) -> None:
        super(SecureNN, self).__init__(
            server_0=server_0 or get_default_config().get_player('server0'),
            server_1=server_1 or get_default_config().get_player('server1'),
            crypto_producer=server_2 or get_default_config().get_player('crypto_producer'),
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
        return self.lsb(x * 2)

    def lsb(self, x: PondTensor) -> PondTensor:
        raise NotImplementedError

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

    def _private_compare_beta0(self, input: PondPrivateTensor, rho: PondPublicTensor):

        w = self.bitwise_xor(input, rho)
        c = rho - input + 1  # + w0_sum

        return c

    def _private_compare_beta1(self, input: PondPrivateTensor, theta: PondPublicTensor):

        w = self.bitwise_xor(input, theta)
        c = input - theta + 1  # + sum

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

            w = self.bitwise_xor(input, rho)

            with tf.name_scope('find_zeros'):
                print('zeros', beta.shape)
                eq = self.equal(beta, 0)
                print('zeros', eq.shape)
                zeros = self.where(eq)
                print('zeros', zeros.shape)
                # return tf.Print(zeros.value_on_0.value, [zeros.value_on_0.value], message="Zeros tf tensor")

            with tf.name_scope('find_ones'):
                eq = self.equal(beta, 1)
                ones = self.where(eq)

            with tf.name_scope('find_edges'):
                edges = self.where(self.equal(rho, 2 ** bits - 1))

            # with tf.name_scope('find_non_edge_ones'):
            #     ones = tf.setdiff1d(ones, edges)

            # # TODO -- needs and equivalent of `take`
            pc_0 = self._private_compare_beta0(input, rho)
            pc_1 = self._private_compare_beta1(input, theta)
            # c0_edge, c1_edge = self._private_compare_edge()

            pc_0 = pc_0.reshape([-1])
            pc_1 = pc_1.reshape([-1])

            with tf.device(self.server_0.device_name):
                c0 = tf.zeros(shape=input.shape, dtype=tf.int32)
                print('pc0', pc_0.shape)
                print('types', zeros.value_on_0.value.shape, pc_0.share0.shape, input)

                delta0 = tf.SparseTensor(zeros.value_on_0.value, pc_0.share0.value, input.shape)
                delta1 = tf.SparseTensor(zeros.value_on_1.value, pc_1.share0.value, input.shape)

                c0 = c0 + tf.sparse_tensor_to_dense(delta0) + tf.sparse_tensor_to_dense(delta1)
                c0 = Int32Tensor(c0)

            with tf.device(self.server_0.device_name):
                c1 = tf.zeros(shape=input.shape, dtype=tf.int32)

                delta0 = tf.SparseTensor(zeros.value_on_0.value, pc_0.share1.value, input.shape)
                delta1 = tf.SparseTensor(zeros.value_on_1.value, pc_1.share1.value, input.shape)

                c1 = c1 + tf.sparse_tensor_to_dense(delta0) + tf.sparse_tensor_to_dense(delta1)
                c1 = Int32Tensor(c1)

            print('hi', c0)
            answer = PondPrivateTensor(self, share0=c0, share1=c1, is_scaled=input.is_scaled)
            return answer

            """
            zero_indices = np.expand_dims(zero_indices, 1)
            one_indices = np.expand_dims(one_indices, 1)
            edge_indices = np.expand_dims(edge_indices, 1)

            np.put_along_axis(c0, zero_indices, c0_zero, axis=0)
            np.put_along_axis(c0, one_indices, c0_one, axis=0)
            np.put_along_axis(c0, edge_indices, c0_edge, axis=0)

            np.put_along_axis(c1, zero_indices, c1_zero, axis=0)
            np.put_along_axis(c1, one_indices, c1_one, axis=0)
            np.put_along_axis(c1, edge_indices, c1_edge, axis=0)
            """

            # # TODO - how to send to the third party? (crypto producer)
            # with tf.device(self.crypto_producer.device_name):
            #     answer = PondPrivateTensor(self, share0=c0, share1=c1).reveal()
            #
            # return answer

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
