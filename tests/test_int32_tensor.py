import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.tensor.int32 import Int32Factory, Int32Tensor


class TestInt32Tensor(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_pond(self) -> None:
        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer'),
                               tensor_factory=Int32Factory(), use_noninteractive_truncation=True,
                               verify_precision=False) as prot:
            x = prot.define_private_variable(np.array([2, 2]), apply_scaling=False)
            y = prot.define_public_variable(np.array([2, 2]), apply_scaling=False)

            z = x * y

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())
                out = z.reveal().eval(sess)
                np.testing.assert_array_almost_equal(out, [4, 4], decimal=3)

    def test_binarize(self) -> None:
        x = Int32Tensor(tf.constant([2**31 + 2**31 + 3, 3], dtype=np.int32))

        y = x.binarize()

        expected = [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        with tf.Session() as sess:
            actual = sess.run(y.value)

        np.testing.assert_array_equal(actual, expected)

    def test_random_binarize(self) -> None:
        input = np.random.uniform(high=2**31 - 1, size=2000).astype('int32').tolist()
        x = Int32Tensor(tf.constant(input, dtype=tf.int32))

        y = x.binarize()

        with tf.Session() as sess:
            actual = sess.run(y.value)

        j = 0
        for i in input:
            binary = bin(i)[2:].zfill(32)[::-1][:31]
            bin_list = np.array(list(binary)).astype('int32')
            np.testing.assert_equal(actual[j], bin_list)
            j += 1


if __name__ == '__main__':
    unittest.main()
