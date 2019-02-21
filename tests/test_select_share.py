import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestSelectShare(unittest.TestCase):

    def test_selectShare(self):

        alice = np.array([1, 1, 1, 1]).astype(np.float32)
        bob = np.array([2, 2, 2, 2]).astype(np.float32)
        bit = np.array([1, 0, 1, 0]).astype(np.float32)

        expected = np.array([2, 1, 2, 1]).astype(np.float32)

        with tfe.protocol.SecureNN() as prot:
            alice_input = prot.define_private_variable(alice, apply_scaling=True)
            bob_input = prot.define_private_variable(bob, apply_scaling=True)
            bit_input = prot.define_private_variable(bit, apply_scaling=False)

            select = prot.select(bit_input, alice_input, bob_input)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                chosen = sess.run(select.reveal())

                np.testing.assert_equal(expected, chosen)


if __name__ == '__main__':
    unittest.main()
