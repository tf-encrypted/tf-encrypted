import tensorflow as tf
import numpy as np

from .protocol import memoize
from ..protocol.pond import (
    Pond, PondTensor, PondPrivateTensor, PondPublicTensor
)
from ..player import Player
from tensorflow_encrypted.tensor.int32 import Int32Factory, Int32Tensor
from tensorflow_encrypted.tensor.native_shared import binarize

bits = 32


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
        with tf.device(self.server_0.device_name):
            wa = 2 * rho * input
            w = input - wa

            w0_sum = tf.zeros(shape=w.shape)
            # for i in range(bits-1, -1, -1):
            # TODO -- subscripting
            # sum = tf.reduce_sum(w[:, i + 1:], axis=1)
            # w0_sum[:, i] = sum % p

            c0 = (input * -1) + w0_sum

        with tf.device(self.server_1.device_name):
            wa = 2 * rho * input
            wb = rho - wa
            w = input + wb

            w1_sum = tf.zeros(shape=w.shape)
            # for i in range(bits-1, -1, -1):
            # TODO -- subscripting
            # sum = tf.reduce_sum(w[:, i + 1:], axis=1)
            # w1_sum[:, i] = sum % p

            c1 = rho + (input * -1) + 1 + w0_sum

        return c0, c1

    def _private_compare_beta1(self, input: PondPrivateTensor, theta: PondPublicTensor):
        with tf.device(self.server_0.device_name):
            wa = 2 * theta * input
            w = input - wa

            w0_sum = tf.zeros(shape=w.shape)
            # for i in range(bits-1, -1, -1):
            # TODO -- subscripting
            # sum = tf.reduce_sum(w[:, i + 1:], axis=1)
            # w0_sum[:, i] = sum % p

            c0 = (input * -1) + w0_sum

        with tf.device(self.server_1.device_name):
            wa = 2 * rho * input
            wb = rho - wa
            w = input + wb

            w1_sum = tf.zeros(shape=w.shape)
            # for i in range(bits-1, -1, -1):
            # TODO -- subscripting
            # sum = tf.reduce_sum(w[:, i + 1:], axis=1)
            # w1_sum[:, i] = sum % p

            c1 = -theta + input + 1 + w0_sum

        return c0, c1

    def _private_compare_edge(self):
        random_values_u = self.tensor_factory.Tensor.sample_random_tensor((bits,), modulus=p - 1) + 1
        c0 = (random_values_u + 1) % p
        c1 = (-random_values_u) % p

        c0[0] = random_values_u[0]

        return c0, c1

    def private_compare(self, input: PondPrivateTensor, rho: PondPublicTensor, beta: PondPublicTensor):
        with tf.name_scope('private_compare'):
            w = self.bitwise_xor(input, rho)
            j = PondPublicTensor(self, value_on_0=Int32Tensor(tf.constant(np.array([1]), dtype=tf.int32)), value_on_1=Int32Tensor(np.array([1])), is_scaled=False)

            zeros = tf.where(beta == 0)[0]
            ones = tf.where(beta == 1)[1]
            edges = tf.where(r == 2 ** bits - 1)[0]
            ones = np.setdiff1d(ones, edges)

            # TODO -- needs and equivalent of `take`
            c0_zeros, c1_zeros = self._private_compare_beta0(input, rho)
            c0_ones, c1_ones = self._private_compare_beta1(input, rho)
            c0_edge, c1_edge = self._private_compare_edge()

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

            # TODO - how to send to the third party? (crypto producer)
            with tf.device(self.server_2.device_name):
                answer = PondPrivateTensor(self, share0=c0, share1=c1).reveal()

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
