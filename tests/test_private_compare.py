import unittest
import random

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.tensor.int32 import int32factory
from tensorflow_encrypted.tensor.prime import PrimeFactory
from tensorflow_encrypted.protocol.pond import PondPrivateTensor, PondPublicTensor

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

    def test_private(self):

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'server2'
        ])

        x = np.array([
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21
        ], dtype=np.int32).reshape(2, 2, 2)

        r = np.array([
            36,
            20,
            21,
            22,
            36,
            20,
            21,
            22
        ], dtype=np.int32).reshape(2, 2, 2)

        beta = np.array([
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1
        ], dtype=np.int32).reshape(2, 2, 2)

        expected = np.bitwise_xor(x > r, beta.astype(bool)).astype(np.int32)

        prime_factory = PrimeFactory(37)

        prot = tfe.protocol.SecureNN(
            tensor_factory=int32factory,
            prime_factory=prime_factory,
            use_noninteractive_truncation=True,
            verify_precision=False,
            *config.get_players('server0, server1, server2'))

        res = prot._private_compare(
            x_bits = PondPrivateTensor(
                prot,
                *prot._share(prime_factory.tensor(tf.convert_to_tensor(x)).to_bits(), prime_factory),
                False),
            r = PondPublicTensor(
                prot,
                prime_factory.tensor(tf.convert_to_tensor(r)),
                prime_factory.tensor(tf.convert_to_tensor(r)),
                False),
            beta = PondPublicTensor(
                prot,
                prime_factory.tensor(tf.convert_to_tensor(beta)),
                prime_factory.tensor(tf.convert_to_tensor(beta)),
                False)
        )

        with config.session() as sess:
            actual = sess.run(res.reveal().value_on_0.value)
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
