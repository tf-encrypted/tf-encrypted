import unittest
import random

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.tensor.int32 import Int32Factory, Int32Tensor
from tensorflow_encrypted.protocol.pond import PondPrivateTensor, PondPublicTensor
from tensorflow_encrypted.tensor.prime import PrimeTensor, prime_factory

bits = 32
Q = 2 ** bits


# Hacky using numpy until we have binarize in PrimeTensor
def binarize(tensor):
    """Computes bit decomposition of tensor
     tensor: ndarray of shape (x0, ..., xn)
    returns: a binary tensor of shape (x0, ..., xn, bits) equivalent to tensor
    """
    bitwidths = np.arange(bits, dtype=np.int32)
    for i in range(len(tensor.shape)):
        bitwidths = np.expand_dims(bitwidths, 0)
    tensor = np.expand_dims(tensor, -1)
    tensor = np.right_shift(tensor, bitwidths) & 1
    return tensor


def sample_random_tensor(shape, modulus=Q):
    if len(shape) >= 1:
        n = np.prod(shape)
    else:
        n = 1
    values = [random.randrange(modulus) for _ in range(n)]
    return np.array(values, dtype=np.int64).reshape(shape)


def share(secrets, modulus=Q):
    shares0 = sample_random_tensor(secrets.shape, modulus)
    shares1 = (secrets - shares0)  # % modulus
    return shares0, shares1


class TestPrivateCompare(unittest.TestCase):

    def test_private_compare_0(self):

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'server2'
        ])

        x = tf.constant([
            [1,0,0,1,1,0,1],
            [1,0,0,1,1,0,1],
            [0,0,0,1,1,0,1],
            [0,0,0,0,0,0,1]], dtype=tf.int32)
        
        r = tf.constant([
            [1,0,0,1,1,0,1],
            [0,0,0,1,1,0,1],
            [1,0,0,1,1,0,1],
            [1,1,1,1,1,1,0]], dtype=tf.int32)

        prot = tfe.protocol.SecureNN(
            tensor_factory=Int32Factory(),
            prime_factory=prime_factory(37),
            use_noninteractive_truncation=True,
            verify_precision=False,
            *config.get_players('server0, server1, server2'))

        x = PondPrivateTensor(prot, *prot._share(PrimeTensor(x, 37), prime_factory(37)), False)
        r = PondPublicTensor(prot, PrimeTensor(r, 37), PrimeTensor(r, 37), False)

        res = prot._private_compare_beta0(x, r)

        with config.session() as sess:
            print(sess.run(res.reveal().value_on_0.value))


    # def test_privateCompare(self):

    #     config = tfe.LocalConfig([
    #         'server0',
    #         'server1',
    #         'crypto_producer'
    #     ])

    #     input = np.array([2, 8, 8, 5, 2, 5, 1, 100]).astype(np.int32)
    #     rho = np.array([1, 7, 7, 4, 1, 2, 0, 99]).astype(np.int32)
    #     beta = np.array( [0, 1, 0, 1, 0, 1, 0, 0]).astype(np.int32)
    #                    # [1, 0, 1, 0, 1, 0, 1, 1]

    #     with tfe.protocol.SecureNN(tensor_factory=Int32Factory(), prime_factory=prime_factory(37), use_noninteractive_truncation=True, verify_precision=False, *config.get_players('server0, server1, crypto_producer')) as prot:

    #         # input = prot.define_private_variable(binarize(input), apply_scaling=False)
    #         # rho = prot.define_public_variable(binarize(rho), apply_scaling=False)
    #         # beta = prot.define_public_variable(binarize(beta), apply_scaling=False)

    #         input = Int32Tensor(input).to_bits()
    #         # theta = binarize(rho + 1)
    #         # rho = binarize(rho)

    #         i_0, i_1 = prot._share(input, factory=prot.prime_factory)

    #         input = PondPrivateTensor(prot, share0=i_0, share1=i_1, is_scaled=False)
    #         rho = PondPublicTensor(prot, value_on_0=Int32Tensor(tf.constant(rho, dtype=tf.int32)),
    #                                value_on_1=Int32Tensor(tf.constant(rho, dtype=tf.int32)), is_scaled=False)
    #         beta = PondPublicTensor(prot, value_on_0=Int32Tensor(tf.constant(beta, dtype=tf.int32)),
    #                                 value_on_1=Int32Tensor(tf.constant(beta, dtype=tf.int32)), is_scaled=False)

    #         #
    #         # i = tf.placeholder(tf.int32)
    #         # r = tf.placeholder(tf.int32)
    #         # b = tf.placeholder(tf.int32)
    #         #

    #         print('inputs', input, rho, beta)
    #         a = prot.private_compare(input, rho, beta)

    #         writer = tf.summary.FileWriter('.')
    #         writer.add_graph(tf.get_default_graph())
    #         #
    #         # eq = prot.equal(beta, 0)
    #         # ones = prot.where(eq)

    #         # sess = tf.Session()
    #         with config.session() as sess:
    #             sess.run(tf.global_variables_initializer())
    #             answer = a.reveal().eval(sess)
    #             # answer = sess.run(a)

    #             print('answer', answer)


if __name__ == '__main__':
    unittest.main()
