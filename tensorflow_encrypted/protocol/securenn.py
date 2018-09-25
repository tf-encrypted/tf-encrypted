import tensorflow as tf
import numpy as np

from .protocol import memoize
from ..protocol.pond import (
    Pond, PondTensor, PondPrivateTensor
)
from ..player import Player
from tensorflow_encrypted.tensor.int32 import Int32Factory, Int32Tensor

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
        w = y - x
        c = bit * w

        return x + c + PondPrivateTensor.zero(x.prot, x.shape)

    def private_compare(self, x, r, beta):

        # TODO -- the broadcasting step for r & beta
        #         the test currently has them in same shape
        #         as the input so it is not needed yet

        # t = (r + 1) % (2 ** bits)
        # print(f'T: {type(t)} ')

        with tf.name_scope('private_compare'):
            r_binary = r.binarize()
            # t_binary = t.binarize()
            input = x.binarize()

            for server in [self.server_0, self.server_1]:
                with tf.device(server.device_name):
                    c = Int32Tensor(np.zeros(shape=input.shape))


        """
        r = np.broadcast_to(np.array(r), tensor.shape[0])
    beta = np.broadcast_to(np.array(beta), tensor.shape[0])

    # t = t = ((r + 1) % -np.iinfo(np.int64).max - 1) + np.iinfo(np.int64).max + 1
    t = (r + 1) % (2 ** bits)

    r_binary = binarize(r)
    t_binary = binarize(t)

    t0 = tensor.shares0
    t1 = tensor.shares1

    c0 = np.zeros(shape=t1.shape)
    c1 = np.zeros(shape=t1.shape)

    zero_indices = np.where(beta == 0)[0]
    one_indices = np.where(beta == 1)[0]
    edge_indices = np.where(r == 2 ** bits - 1)[0]
    one_indices = np.setdiff1d(one_indices, edge_indices)

    c0_zero, c1_zero = _private_compare_beta0(r_binary.take(zero_indices, axis=0),
                                              t0.take(zero_indices, axis=0),
                                              t1.take(zero_indices, axis=0))
    c0_one, c1_one = _private_compare_beta1(r_binary.take(one_indices, axis=0),
                                            t_binary.take(one_indices, axis=0),
                                            t0.take(one_indices, axis=0),
                                            t1.take(one_indices, axis=0))
    c0_edge, c1_edge = _private_compare_edge()

    zero_indices = np.expand_dims(zero_indices, 1)
    one_indices = np.expand_dims(one_indices, 1)
    edge_indices = np.expand_dims(edge_indices, 1)

    np.put_along_axis(c0, zero_indices, c0_zero, axis=0)
    np.put_along_axis(c0, one_indices, c0_one, axis=0)
    np.put_along_axis(c0, edge_indices, c0_edge, axis=0)

    np.put_along_axis(c1, zero_indices, c1_zero, axis=0)
    np.put_along_axis(c1, one_indices, c1_one, axis=0)
    np.put_along_axis(c1, edge_indices, c1_edge, axis=0)

    res = PrivateTensor(values=None, shares0=c0, shares1=c1, modulus=p).reconstruct().values

    return np.max(res == 0, axis=-1)
    """
        return 0

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
