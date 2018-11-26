import unittest

from typing import List

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestPad(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_pad(self):

        with tfe.protocol.Pond() as prot:

            tf.reset_default_graph()

            input = np.array([[1, 2, 3], [4, 5, 6]])
            input_input = prot.define_private_variable(input)

            paddings = [[2, 2], [3, 4]]

            out = prot.pad(input_input, paddings)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                out_tfe = sess.run(out.reveal())

            tf.reset_default_graph()

            out_tensorflow = run_pad([2, 3])

            np.testing.assert_allclose(out_tfe, out_tensorflow, atol=.01)


def run_pad(input_shape: List[int]):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")

    x = tf.pad(a, paddings=tf.constant([[2, 2], [3, 4]]), mode="CONSTANT")

    with tf.Session() as sess:
        output = sess.run(x, feed_dict={a: np.array([[1, 2, 3], [4, 5, 6]])})

    return output


if __name__ == '__main__':
    unittest.main()
