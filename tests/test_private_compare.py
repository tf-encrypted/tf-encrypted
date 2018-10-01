import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
import random

from tensorflow_encrypted.tensor.int32 import Int32Factory, Int32Tensor
from tensorflow_encrypted.protocol.pond import PondPrivateTensor, PondPublicTensor
from tensorflow_encrypted.tensor.prime import prime_factory

bits = 32
Q = 2 ** bits


# Hacky using numpy until we have binarize in PrimeTensor
def binarize(tensor):
    """Computes bit decomposition of tensor
     tensor: ndarray of shape (x0, ..., xn)
    returns: a binary tensor of shape (x0, ..., xn, bits) equivalent to tensor
    """
    bitwidths = np.arange(bits, dtype=np.int64)
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
    shares1 = (secrets - shares0) % modulus
    return shares0, shares1


class TestPrivateCompare(unittest.TestCase):

    def test_privateCompare(self):

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        input = np.array([1, 1, 1, 1]).astype(np.int32)
        rho = np.array([2, 0, 2, 0]).astype(np.int32)
        beta = np.array([0, 0, 0, 0]).astype(np.int32)

        with tfe.protocol.SecureNN(tensor_factory=prime_factory(2 ** 31), use_noninteractive_truncation=True, verify_precision=False, *config.get_players('server0, server1, crypto_producer')) as prot:

            input = prot.define_private_variable(binarize(input), apply_scaling=False)
            rho = prot.define_public_variable(binarize(rho), apply_scaling=False)
            beta = prot.define_public_variable(binarize(beta), apply_scaling=False)

            # input = PondPrivateTensor(prot, share0=Int32Tensor(tf.constant(i_0, dtype=tf.int32)), share1=Int32Tensor(tf.constant(i_1, dtype=tf.int32)), is_scaled=False)
            # rho = PondPublicTensor(prot, value_on_0=Int32Tensor(tf.constant(rho, dtype=tf.int32)), value_on_1=Int32Tensor(tf.constant(rho, dtype=tf.int32)), is_scaled=False)
            # beta = PondPublicTensor(prot, value_on_0=Int32Tensor(tf.constant(beta, dtype=tf.int32)), value_on_1=Int32Tensor(tf.constant(beta, dtype=tf.int32)), is_scaled=False)

            #
            # i = tf.placeholder(tf.int32)
            # r = tf.placeholder(tf.int32)
            # b = tf.placeholder(tf.int32)
            #

            a = prot.private_compare(input, rho, beta)

            print('returned', a)

            writer = tf.summary.FileWriter('.')
            writer.add_graph(tf.get_default_graph())

            # sess = tf.Session()
            with config.session() as sess:
                sess.run(tf.global_variables_initializer())
                answer = a.reveal().eval(sess)
                # answer = sess.run(a)

                print('answer', answer)

                # sess.run(answer, feed_dict=)

            # print(res)

            #
            # input = Int32Tensor(input)
            # beta = Int32Tensor(beta)
            # r = Int32Tensor(r)
            #
            # a, b, c = prot.private_compare(input, r, beta)
            #
            # writer = tf.summary.FileWriter('.')
            # writer.add_graph(tf.get_default_graph())
            #
            # with config.session() as sess:
            #     sess.run(tf.global_variables_initializer())
            #     sess.run(a, feed_dict={})
            #     # answer = a.eval(sess, feed_dict={})
            #     print(f'answer: {answer}')
                # chosen = compare.reveal().eval(sess)
                #
                # assert(np.array_equal(expected, compare))


if __name__ == '__main__':
    unittest.main()
