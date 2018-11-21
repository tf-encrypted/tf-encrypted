import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestPad(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_zeros(self):

        with tfe.protocol.Pond() as prot:

            tf.reset_default_graph()

            input = np.array([[1, 2, 3], [4, 5, 6]])
            input_input = prot.define_private_variable(input)

            paddings = [[1, 1], [3, 4]]

            out =  prot.pad(input_input, paddings)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                out_tfe = sess.run(out.reveal())

            tf.reset_default_graph()
        
            t = tf.constant([[1, 2, 3], [4, 5, 6]], dtype='float32')
            paddings = tf.constant([[1, 1], [3, 4]])
            pad_out_tf = tf.pad(t, paddings, constant_values=0)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                out_tensorflow = sess.run(pad_out_tf)

            np.testing.assert_allclose(out_tfe, out_tensorflow, atol=.01)


if __name__ == '__main__':
    unittest.main()