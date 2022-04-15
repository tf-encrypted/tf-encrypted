
# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.operations import aux

disabled_msg = "Aux module disabled"
dontskip = (aux.aux_module is not None)


@unittest.skipUnless(dontskip, disabled_msg)
class TestAux(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def tearDown(self):
        tf.reset_default_graph()

    def test_bit_gather1(self):
        x = tf.constant([0xaaaa], dtype=tf.int64)
        y = aux.bit_gather(x, 0, 2)
        z = aux.bit_gather(x, 1, 2)

        with tf.Session():
            y = y.eval()
            print("y: ", y)
            np.testing.assert_array_equal(y, np.array([0]))
            z = z.eval()
            print("z: ", z)
            np.testing.assert_array_equal(z, np.array([0xff]))

    def test_bit_gather2(self):
        x = tf.constant([0x425f32ea92], dtype=tf.int64)
        y = aux.bit_gather(x, 0, 2)
        z = aux.bit_gather(x, 1, 2)

        with tf.Session():
            y = y.eval()
            np.testing.assert_array_equal(y, np.array([0x8f484]))
            z = z.eval()
            np.testing.assert_array_equal(z, np.array([0x135f9]))


    def test_bit_split_and_gather(self):
        x = tf.constant([0xaaaa, 0x425f32ea92, 0x2], dtype=tf.int64)
        y = aux.bit_split_and_gather(x, 2)
        assert list(y.shape) == [2, 3]

        with tf.Session():
            y = y.eval()
            np.testing.assert_array_equal(y, np.array([[0, 0x8f484, 0], [0xff, 0x135f9, 0x1]]))


if __name__ == "__main__":
    unittest.main()

