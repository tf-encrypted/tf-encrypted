import unittest
import tensorflow_encrypted as tfe
import tensorflow as tf
import numpy as np
from tensorflow_encrypted.tensor.prime import PrimeFactory
from tensorflow_encrypted.protocol.pond import PondPublicTensor
from tensorflow_encrypted.layers.pooling import MaxPooling2D


class TestMaxPooling2D(unittest.TestCase):

    # def test_maxpool2d_secureNN_only(self):
    #     with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:
    #         pool = MaxPooling2D

    def test_maxpool2d(self):
        bit_dtype = PrimeFactory(37)
        # val_dtype = int32factory
        val_dtype = bit_dtype

        prot = tfe.protocol.SecureNN(
            tensor_factory=val_dtype,
            prime_factory=bit_dtype,
            use_noninteractive_truncation=True,
            verify_precision=False
        )

        tfe.set_protocol(prot)

        input = np.array([[[[1, 2, 3, 4], [3, 2, 4, 1], [1, 2, 3, 4], [3, 2, 4, 1]]]])
        input = tf.convert_to_tensor(input)

        input = PondPublicTensor(prot, val_dtype.tensor(input), val_dtype.tensor(input), False)
        pool = MaxPooling2D([0, 1, 4, 4], pool_size=2, padding="VALID")
        result = pool.forward(input)

        print('answer', result)


if __name__ == '__main__':
    unittest.main()
