import unittest

import tensorflow as tf
import numpy as np


shared_object = '../../tf_encrypted/operations/secure_random/secure_random.so'


class TestSecureRandom(unittest.TestCase):

    def test_int32_return(self):
        secure_random_module = tf.load_op_library(shared_object)

        expected = [[-1178104854, 1419540550, -971089687], [432654542, -733903376, 1856463548]]

        with tf.Session():
            output = secure_random_module.secure_random([2, 3], [1, 1, 1, 1, 1, 1, 1, 1]).eval()

            np.testing.assert_array_equal(output, expected)


if __name__ == '__main__':
    unittest.main()
