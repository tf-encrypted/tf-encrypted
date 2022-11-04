# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.operations import aux

disabled_msg = "Aux module disabled"
dontskip = aux.aux_module is not None


@unittest.skipUnless(dontskip, disabled_msg)
class TestAux(unittest.TestCase):
    def test_bit_gather1(self):
        x = tf.constant([0xAAAA], dtype=tf.int64)
        y = aux.bit_gather(x, 0, 2)
        z = aux.bit_gather(x, 1, 2)

        np.testing.assert_array_equal(y, np.array([0]))
        np.testing.assert_array_equal(z, np.array([0xFF]))

    def test_bit_gather2(self):
        x = tf.constant([0x425F32EA92], dtype=tf.int64)
        y = aux.bit_gather(x, 0, 2)
        z = aux.bit_gather(x, 1, 2)

        np.testing.assert_array_equal(y, np.array([0x8F484]))
        np.testing.assert_array_equal(z, np.array([0x135F9]))

    def test_bit_split_and_gather(self):
        x = tf.constant([0xAAAA, 0x425F32EA92, 0x2], dtype=tf.int64)
        y = aux.bit_split_and_gather(x, 2)
        assert list(y.shape) == [2, 3]

        np.testing.assert_array_equal(
            y, np.array([[0, 0x8F484, 0], [0xFF, 0x135F9, 0x1]])
        )


if __name__ == "__main__":
    unittest.main()
