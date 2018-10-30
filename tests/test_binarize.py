import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.protocol.pond import PondPrivateTensor, PondPublicTensor
from tf_encrypted.protocol.securenn import _private_compare
from tf_encrypted.tensor.int64 import Int64Tensor


class TestPrivateCompare(unittest.TestCase):

    def test_private_negative(self):
        x = Int64Tensor(tf.constant(np.array([-1, -2, -3, 4])))
        x_bits = x.to_bits().to_native()

        y = Int64Tensor(tf.constant(np.array([-1, -2, -3, -4])))
        # y_bits = y.to_bits()  # x.to_bits()

        prot = tfe.protocol.SecureNN()

        bit_dtype = prot.prime_factory
        val_dtype = prot.tensor_factory

        with tfe.Session() as sess:
            sess.run(tf.global_variables_initializer())
            answer = sess.run([x_bits])
            print('answer', answer)


if __name__ == '__main__':
    unittest.main()
