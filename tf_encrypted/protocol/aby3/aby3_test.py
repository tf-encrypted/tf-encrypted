# pylint: disable=all
# pylint: disable=missing-docstring
# flake8: noqa

import os
import math
import tempfile
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.protocol.aby3 import ABY3, ShareType


class TestABY3(unittest.TestCase):

    def test_define_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        @tfe.local_computation("input-provider")
        def provide_input():
            return tf.ones(shape=(2, 2)) * 1.3

        # define inputs
        x = tfe.define_private_variable(tf.ones(shape=(2, 2)))
        y = tfe.define_private_variable(np.ones(shape=(2, 2)))
        z = provide_input()

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(x.reveal())
            np.testing.assert_allclose(result, np.array([[1, 1], [1, 1]]), rtol=0.0, atol=0.01)

            result = sess.run(y.reveal())
            np.testing.assert_allclose(result, np.array([[1, 1], [1, 1]]), rtol=0.0, atol=0.01)

            result = sess.run(z.reveal())
            np.testing.assert_allclose(result, np.array([[1.3, 1.3], [1.3, 1.3]]), rtol=0.0, atol=0.01)

    def test_add_private_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        @tfe.local_computation("input-provider")
        def provide_input():
            return tf.ones(shape=(2, 2)) * 1.3

        # define inputs
        x = tfe.define_private_variable(tf.ones(shape=(2, 2)))
        y = provide_input()

        # define computation
        z = x + y

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            # Should be [[2.3, 2.3], [2.3, 2.3]]
            expected = np.array([[2.3, 2.3], [2.3, 2.3]])
            np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.01)

    def test_add_private_public(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        # define inputs
        x = tfe.define_private_variable(tf.ones(shape=(2, 2)))
        y = tfe.define_constant(np.array([[0.6, 0.7], [0.8, 0.9]]))

        # define computation
        z = x + y
        z = y + z

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            expected = np.array([[2.2, 2.4], [2.6, 2.8]])
            np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.01)

    def test_sub_private_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        @tfe.local_computation("input-provider")
        def provide_input():
            return tf.ones(shape=(2, 2)) * 1.3

        x = tfe.define_private_variable(tf.ones(shape=(2, 2)))
        y = provide_input()

        z = x - y

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            expected = np.array([[-0.3, -0.3], [-0.3, -0.3]])
            np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.01)

    def test_sub_private_public(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        # define inputs
        x = tfe.define_private_variable(tf.ones(shape=(2, 2)))
        y = tfe.define_constant(np.array([[0.6, 0.7], [0.8, 0.9]]))

        # define computation
        z1 = x - y
        z2 = y - x

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            z1_exp = np.array([[0.4, 0.3], [0.2, 0.1]])
            np.testing.assert_allclose(result, z1_exp, rtol=0.0, atol=0.01)
            result = sess.run(z2.reveal())
            z2_exp = np.array([[-0.4, -0.3], [-0.2, -0.1]])
            np.testing.assert_allclose(result, z2_exp, rtol=0.0, atol=0.01)

    def test_neg(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        # define inputs
        x = tfe.define_private_variable(np.array([[0.6, -0.7], [-0.8, 0.9]]))
        y = tfe.define_constant(np.array([[0.6, -0.7], [-0.8, 0.9]]))

        # define computation
        z1 = -x
        z2 = -y

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            z1_exp = np.array([[-0.6, 0.7], [0.8, -0.9]])
            np.testing.assert_allclose(result, z1_exp, rtol=0.0, atol=0.01)
            result = sess.run(z2)
            z2_exp = np.array([[-0.6, 0.7], [0.8, -0.9]])
            np.testing.assert_allclose(result, z2_exp, rtol=0.0, atol=0.01)

    def test_mul_private_public(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        # define inputs
        x = tfe.define_private_variable(tf.ones(shape=(2, 2)) * 2)
        y = tfe.define_constant(np.array([[0.6, 0.7], [0.8, 0.9]]))
        w = tfe.define_constant(np.array([[2, 2], [2, 2]]))

        # define computation
        z1 = y * x  # mul_public_private
        z2 = z1 * w  # mul_private_public

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z2.reveal())
            np.testing.assert_allclose(
                result, np.array([[2.4, 2.8], [3.2, 3.6]]), rtol=0.0, atol=0.01
            )

    def test_mul_private_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        @tfe.local_computation("input-provider")
        def provide_input():
            # normal TensorFlow operations can be run locally
            # as part of defining a private input, in this
            # case on the machine of the input provider
            return tf.ones(shape=(2, 2)) * 1.3

        # define inputs
        x = tfe.define_private_variable(tf.ones(shape=(2, 2)) * 2)
        y = provide_input()

        # define computation
        z = y * x

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            np.testing.assert_allclose(
                result, np.array([[2.6, 2.6], [2.6, 2.6]]), rtol=0.0, atol=0.01
            )

    def test_matmul_public_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        @tfe.local_computation("input-provider")
        def provide_input():
            # normal TensorFlow operations can be run locally
            # as part of defining a private input, in this
            # case on the machine of the input provider
            return tf.constant(np.array([[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]))

        # define inputs
        x = tfe.define_private_variable(tf.ones(shape=(2, 2)))
        y = provide_input()
        v = tfe.define_constant(np.ones((2, 2)))

        # define computation
        w = y.matmul(x)  # matmul_public_private
        z = w.matmul(v)  # matmul_private_public

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(w.reveal())
            np.testing.assert_allclose(
                result,
                np.array([[2.3, 2.3], [2.7, 2.7], [3.1, 3.1]]),
                rtol=0.0,
                atol=0.01,
            )
            result = sess.run(z.reveal())
            np.testing.assert_allclose(
                result,
                np.array([[4.6, 4.6], [5.4, 5.4], [6.2, 6.2]]),
                rtol=0.0,
                atol=0.01,
            )

    def test_matmul_private_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        # 2-D matrix mult
        x = tfe.define_private_variable(tf.constant([[1, 2, 3], [4, 5, 6]]))
        y = tfe.define_private_variable(tf.constant([[7, 8], [9, 10], [11, 12]]))

        z = tfe.matmul(x, y)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            np.testing.assert_allclose(
                result, np.array([[58, 64], [139, 154]]), rtol=0.0, atol=0.01
            )

    def test_3d_matmul_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        # 3-D matrix mult
        x = tfe.define_private_variable(tf.constant(np.arange(1, 13), shape=[2, 2, 3]))
        y = tfe.define_private_variable(tf.constant(np.arange(13, 25), shape=[2, 3, 2]))

        z = tfe.matmul(x, y)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            np.testing.assert_allclose(
                result,
                np.array([[[94, 100], [229, 244]], [[508, 532], [697, 730]]]),
                rtol=0.0,
                atol=0.01,
            )

    def test_cast(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        # define inputs
        x = tfe.define_private_variable(np.array([[0.6, -0.7], [-0.8, 0.9]]))
        y = tfe.define_constant(np.array([[0.6, -0.7], [-0.8, 0.9]]))

        # define computation
        z1 = x.cast(prot.factories[tf.int32])
        z2 = y.cast(prot.factories[tf.int32])

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(result, np.array([[0.6, -0.7], [-0.8, 0.9]]), rtol=0.0, atol=0.01)
            result = sess.run(z2)
            np.testing.assert_allclose(result, np.array([[0.6, -0.7], [-0.8, 0.9]]), rtol=0.0, atol=0.01)

    def test_boolean_sharing(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            tf.constant([[1, 2, 3], [4, 5, 6]]), share_type=ShareType.BOOLEAN
        )
        y = tfe.define_private_variable(
            tf.constant([[7, 8, 9], [10, 11, 12]]), share_type=ShareType.BOOLEAN
        )

        z1 = tfe.xor(x, y)

        z2 = tfe.and_(x, y)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[6, 10, 10], [14, 14, 10]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z2.reveal())
            np.testing.assert_allclose(
                result, np.array([[1, 0, 1], [0, 1, 4]]), rtol=0.0, atol=0.01
            )

    def test_not_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            tf.constant([[1, 2, 3], [4, 5, 6]]), share_type=ShareType.BOOLEAN, apply_scaling=False
        )
        y = tfe.define_private_variable(
            tf.constant([[1, 0, 0], [0, 1, 0]]),
            apply_scaling=False,
            share_type=ShareType.BOOLEAN,
            factory=prot.factories[tf.bool],
        )
        z1 = ~x
        z2 = ~y

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[-2, -3, -4], [-5, -6, -7]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z2.reveal())
            np.testing.assert_allclose(
                result, np.array([[0, 1, 1], [1, 0, 1]]), rtol=0.0, atol=0.01
            )

    def test_native_ppa_sklansky(self):
        from math import log2
        from random import randint

        n = 10
        while n > 0:
            n = n - 1

            x = randint(1, 2 ** 31)
            y = randint(1, 2 ** 31)
            keep_masks = [
                0x5555555555555555,
                0x3333333333333333,
                0x0F0F0F0F0F0F0F0F,
                0x00FF00FF00FF00FF,
                0x0000FFFF0000FFFF,
                0x00000000FFFFFFFF,
            ]  # yapf: disable
            copy_masks = [
                0x5555555555555555,
                0x2222222222222222,
                0x0808080808080808,
                0x0080008000800080,
                0x0000800000008000,
                0x0000000080000000,
            ]  # yapf: disable

            G = x & y
            P = x ^ y
            k = 64
            for i in range(int(log2(k))):
                c_mask = copy_masks[i]
                k_mask = keep_masks[i]
                # Copy the selected bit to 2^i positions:
                # For example, when i=2, the 4-th bit is copied to the (5, 6, 7, 8)-th bits
                G1 = (G & c_mask) << 1
                P1 = (P & c_mask) << 1
                for j in range(i):
                    G1 = (G1 << (2 ** j)) ^ G1
                    P1 = (P1 << (2 ** j)) ^ P1
                # Two-round impl. using algo. specified in the slides that assume using OR gate is free, but in fact,
                # here using OR gate cost one round.
                # The PPA operator 'o' is defined as:
                # (G, P) o (G1, P1) = (G + P*G1, P*P1), where '+' is OR, '*' is AND

                # G1 and P1 are 0 for those positions that we do not copy the selected bit to.
                # Hence for those positions, the result is: (G, P) = (G, P) o (0, 0) = (G, 0).
                # In order to keep (G, P) for these positions so that they can be used in the future,
                # we need to let (G1, P1) = (G, P) for these positions, because (G, P) o (G, P) = (G, P)

                # G1 = G1 ^ (G & k_mask)
                # P1 = P1 ^ (P & k_mask)

                # G = G | (P & G1)
                # P = P & P1

                # One-round impl. by modifying the PPA operator 'o' as:
                # (G, P) o (G1, P1) = (G ^ (P*G1), P*P1), where '^' is XOR, '*' is AND
                # This is a valid definition: when calculating the carry bit c_i = g_i + p_i * c_{i-1},
                # the OR '+' can actually be replaced with XOR '^' because we know g_i and p_i will NOT take '1'
                # at the same time.
                # And this PPA operator 'o' is also associative. BUT, it is NOT idempotent, hence (G, P) o (G, P) != (G, P).
                # This does not matter, because we can do (G, P) o (0, P) = (G, P), or (G, P) o (0, 1) = (G, P)
                # if we want to keep G and P bits.

                # Option 1: Using (G, P) o (0, P) = (G, P)
                # P1 = P1 ^ (P & k_mask)
                # Option 2: Using (G, P) o (0, 1) = (G, P)
                P1 = P1 ^ k_mask

                G = G ^ (P & G1)
                P = P & P1

            # G stores the carry-in to the next position
            C = G << 1
            P = x ^ y
            z = C ^ P

            truth = x + y

            assert z == truth

    def test_native_ppa_kogge_stone(self):
        from math import log2
        from random import randint

        n = 10
        while n > 0:
            n = n - 1
            x = randint(1, 2 ** 31)
            y = randint(1, 2 ** 31)
            G = x & y
            P = x ^ y
            keep_masks = [
                0x0000000000000001,
                0x0000000000000003,
                0x000000000000000F,
                0x00000000000000FF,
                0x000000000000FFFF,
                0x00000000FFFFFFFF,
            ]  # yapf: disable
            k = 64
            for i in range(int(log2(k))):
                k_mask = keep_masks[i]
                # Copy the selected bit to 2^i positions:
                # For example, when i=2, the 4-th bit is copied to the (5, 6, 7, 8)-th bits
                G1 = G << (2 ** i)
                P1 = P << (2 ** i)

                # One-round impl. by modifying the PPA operator 'o' as:
                # (G, P) o (G1, P1) = (G ^ (P*G1), P*P1), where '^' is XOR, '*' is AND
                # This is a valid definition: when calculating the carry bit c_i = g_i + p_i * c_{i-1},
                # the OR '+' can actually be replaced with XOR '^' because we know g_i and p_i will NOT take '1'
                # at the same time.
                # And this PPA operator 'o' is also associative. BUT, it is NOT idempotent, hence (G, P) o (G, P) != (G, P).
                # This does not matter, because we can do (G, P) o (0, P) = (G, P), or (G, P) o (0, 1) = (G, P)
                # if we want to keep G and P bits.

                # Option 1: Using (G, P) o (0, P) = (G, P)
                # P1 = P1 ^ (P & k_mask)
                # Option 2: Using (G, P) o (0, 1) = (G, P)
                P1 = P1 ^ k_mask

                G = G ^ (P & G1)
                P = P & P1

            # G stores the carry-in to the next position
            C = G << 1
            P = x ^ y
            z = C ^ P

            truth = x + y

            assert z == truth

    def test_bit_gather(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        # define inputs
        x = tfe.define_private_variable(np.array([0xaaaa, 0x425f32ea92]), apply_scaling=False, share_type=ShareType.BOOLEAN)

        # define computation
        z1 = tfe.bit_gather(x, 0, 2)
        z2 = tfe.bit_gather(x, 1, 2)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_array_equal(result, np.array([0, 0x8f484]))
            result = sess.run(z2.reveal())
            np.testing.assert_array_equal(result, np.array([0xff, 0x135f9]))


    def test_carry(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            tf.constant([[1, -2**63, -1], [-1, -2, 4]], dtype=tf.int64), apply_scaling=False, share_type=ShareType.BOOLEAN
        )
        y1 = tfe.define_private_variable(
            tf.constant([[7, -2**63, 1], [-1, 1, -1]], dtype=tf.int64), apply_scaling=False, share_type=ShareType.BOOLEAN
        )
        y2 = tfe.define_constant(
            np.array([[7, -2**63, 1], [-1, 1, -1]], dtype=np.int64), apply_scaling=False
        )

        z1 = tfe.carry(x, y1).cast(prot.factories[tf.int8])
        z2 = tfe.carry(x, y2).cast(prot.factories[tf.int8])

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_array_equal(
                result, np.array([[0, 1, 1], [1, 0, 1]])
            )

            result = sess.run(z2.reveal())
            np.testing.assert_array_equal(
                result, np.array([[0, 1, 1], [1, 0, 1]])
            )

    def test_lshift_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            tf.constant([[1, 2, 3], [4, 5, 6]]), share_type=ShareType.BOOLEAN
        )

        z = x << 1

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            np.testing.assert_allclose(
                result, np.array([[2, 4, 6], [8, 10, 12]]), rtol=0.0, atol=0.01
            )

    def test_rshift_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            tf.constant([[1, 2, 3], [4, 5, 6]]), share_type=ShareType.BOOLEAN
        )
        y = tfe.define_private_variable(
            tf.constant([[-1, -2, -3], [-4, 5, 6]]),
            share_type=ShareType.BOOLEAN,
            apply_scaling=False,
        )

        z = x >> 1
        w = y >> 1
        s = y.logical_rshift(1)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            np.testing.assert_allclose(
                result,
                np.array(
                    [[0.5, 1, 1.5], [2, 2.5, 3]]
                ),  # NOTE: x is scaled and treated as fixed-point number
                rtol=0.0,
                atol=0.01,
            )
            result = sess.run(w.reveal())
            np.testing.assert_allclose(
                result, np.array([[-1, -1, -2], [-2, 2, 3]]), rtol=0.0, atol=0.01
            )
            result = sess.run(s.reveal())
            np.testing.assert_allclose(
                result,
                np.array(
                    [
                        [
                            (-1 & ((1 << prot.default_nbits) - 1)) >> 1,
                            (-2 & ((1 << prot.default_nbits) - 1)) >> 1,
                            (-3 & ((1 << prot.default_nbits) - 1)) >> 1,
                        ],
                        [(-4 & ((1 << prot.default_nbits) - 1)) >> 1, 2, 3],
                    ]
                ),
                rtol=0.0,
                atol=0.01,
            )

    def test_ppa_private_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            tf.constant([[1, 2, 3], [4, 5, 6]]), share_type=ShareType.BOOLEAN
        )
        y = tfe.define_private_variable(
            tf.constant([[7, 8, 9], [10, 11, 12]]), share_type=ShareType.BOOLEAN
        )

        # Parallel prefix adder. It is simply an adder for boolean sharing.
        z1 = tfe.ppa(x, y, topology="sklansky")
        z2 = tfe.ppa(x, y, topology="kogge_stone")

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[8, 10, 12], [14, 16, 18]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z2.reveal())
            np.testing.assert_allclose(
                result, np.array([[8, 10, 12], [14, 16, 18]]), rtol=0.0, atol=0.01
            )

    def test_a2b_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            tf.constant([[1, 2, 3], [4, 5, 6]]), share_type=ShareType.ARITHMETIC
        )

        z = tfe.a2b(x)
        assert z.share_type == ShareType.BOOLEAN

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            np.testing.assert_allclose(
                result, np.array([[1, 2, 3], [4, 5, 6]]), rtol=0.0, atol=0.01
            )

    def test_b2a_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            tf.constant([[1, 2, 3], [4, 5, 6]]), share_type=ShareType.BOOLEAN
        )

        z = tfe.b2a(x)
        assert z.share_type == ShareType.ARITHMETIC

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            np.testing.assert_allclose(
                result, np.array([[1, 2, 3], [4, 5, 6]]), rtol=0.0, atol=0.01
            )

    def test_b2a_single_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant([[0, 1, 0], [1, 0, 1]]), share_type = ShareType.BOOLEAN, apply_scaling = False, factory = prot.factories[tf.bool])
        y = tfe.b2a_single(x)
        assert y.share_type == ShareType.ARITHMETIC

        with  tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(y.reveal())
            np.testing.assert_allclose(result, np.array([[0, 1, 0], [1, 0, 1]]), rtol=0.0, atol=0.01)

    def test_truncate_heuristic(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        scale = prot.fixedpoint_config.precision_fractional
        a = tf.random_uniform([1], 0, 2**30) # Plus the 15-bit fractional scale, this will be encoded to 45-bit numbers
        x = tfe.define_private_input("server0", lambda: a)
        x = x << scale

        y1 = tfe.truncate(x, method="heuristic")

        with tfe.Session() as sess:
            sess.run(tfe.global_variables_initializer())
            n = 1000
            for i in range(n):
                result1, truth = sess.run(
                    [y1.reveal(), a])
                np.testing.assert_allclose(result1, truth, rtol=0.0, atol=0.1)


    def test_truncate_msb0_cheetah(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        scale = prot.fixedpoint_config.precision_fractional
        x = tfe.define_private_variable(tf.constant([[1, 2, 3], [4, 5, 6]]), share_type = ShareType.ARITHMETIC)
        y = x << scale

        z = tfe.truncate_msb0(y, method="cheetah")

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(y.reveal())
            np.testing.assert_allclose(result, np.array([[1*(2**scale), 2*(2**scale), 3*(2**scale)], [4*(2**scale), 5*(2**scale), 6*(2**scale)]]), rtol=0.0001, atol=0.0001*(2**scale))

            result = sess.run(z.reveal())
            np.testing.assert_allclose(result, np.array([[1, 2, 3], [4, 5, 6]]), rtol=0.00, atol=0.001)

    def test_truncate_msb0_secureq8(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        scale = prot.fixedpoint_config.precision_fractional
        x = tfe.define_private_variable(tf.constant([[1, 2, 3], [4, 5, 6]]), share_type = ShareType.ARITHMETIC)
        y = x << scale

        z = tfe.truncate_msb0(y, method="secureq8")

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(y.reveal())
            np.testing.assert_allclose(result, np.array([[1*(2**scale), 2*(2**scale), 3*(2**scale)], [4*(2**scale), 5*(2**scale), 6*(2**scale)]]), rtol=0.0001, atol=0.0001*(2**scale))

            result = sess.run(z.reveal())
            np.testing.assert_allclose(result, np.array([[1, 2, 3], [4, 5, 6]]), rtol=0.00, atol=0.001)

    def test_ot(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        m0 = prot.define_constant(
            np.array([[1, 2, 3], [4, 5, 6]]), apply_scaling=False
        ).unwrapped[0]
        m1 = prot.define_constant(
            np.array([[2, 3, 4], [5, 6, 7]]), apply_scaling=False
        ).unwrapped[0]
        c_on_receiver = prot.define_constant(
            np.array([[1, 0, 1], [0, 1, 0]]),
            apply_scaling=False,
            factory=prot.factories[tf.bool],
        ).unwrapped[0]
        c_on_helper = prot.define_constant(
            np.array([[1, 0, 1], [0, 1, 0]]),
            apply_scaling=False,
            factory=prot.factories[tf.bool],
        ).unwrapped[0]

        m_c = prot._ot(  # pylint: disable=protected-access
            prot.servers[1],
            prot.servers[2],
            prot.servers[0],
            m0,
            m1,
            c_on_receiver,
            c_on_helper,
            prot.pairwise_keys[1][0],
            prot.pairwise_keys[0][1],
            prot.pairwise_nonces[0],
        )

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(
                prot._decode(m_c, False)
            )  # pylint: disable=protected-access
            np.testing.assert_allclose(
                result, np.array([[2, 2, 4], [4, 6, 6]]), rtol=0.0, atol=0.01
            )

    def test_mul_ab_public_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_constant(np.array([[1, 2, 3], [4, 5, 6]]))
        y = tfe.define_private_variable(
            tf.constant([[1, 0, 0], [0, 1, 0]]),
            apply_scaling=False,
            share_type=ShareType.BOOLEAN,
            factory=prot.factories[tf.bool],
        )

        z = tfe.mul_ab(x, y)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            np.testing.assert_allclose(
                result, np.array([[1, 0, 0], [0, 5, 0]]), rtol=0.0, atol=0.01
            )

    def test_mul_ab_private_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            np.array([[1, 2, 3], [4, 5, 6]]), share_type=ShareType.ARITHMETIC,
        )
        y = tfe.define_private_variable(
            tf.constant([[1, 0, 0], [0, 1, 0]]),
            apply_scaling=False,
            share_type=ShareType.BOOLEAN,
            factory=prot.factories[tf.bool],
        )

        z = tfe.mul_ab(x, y)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            np.testing.assert_allclose(
                result, np.array([[1, 0, 0], [0, 5, 0]]), rtol=0.0, atol=0.01
            )

    def test_bit_extract(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            np.array([[1, -2, 3], [-4, -5, 6]]), share_type=ShareType.ARITHMETIC,
        )
        y = tfe.define_private_variable(
            np.array([[1, -2, 3], [-4, -5, 6]]),
            share_type=ShareType.ARITHMETIC,
            apply_scaling=False,
        )

        z = tfe.bit_extract(
            x, 63
        )  # The sign bit. Since x is scaled, you should be more careful about extracting other bits.
        w = tfe.bit_extract(y, 1)  # y is not scaled

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            np.testing.assert_allclose(
                result.astype(int),
                np.array([[0, 1, 0], [1, 1, 0]]),
                rtol=0.0,
                atol=0.01,
            )
            result = sess.run(w.reveal())
            np.testing.assert_allclose(
                result.astype(int),
                np.array([[0, 1, 1], [0, 1, 1]]),
                rtol=0.0,
                atol=0.01,
            )


    def test_msb(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            np.array([[1, -2, 3], [-4, -5, 6]]), share_type=ShareType.ARITHMETIC,
        )

        s = tfe.msb(x)  # Sign bit

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(s.reveal())
            np.testing.assert_allclose(
                result.astype(int),
                np.array([[0, 1, 0], [1, 1, 0]]),
                rtol=0.0,
                atol=0.01,
            )

    def test_equal_zero(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(np.array([[1, -2, 0], [-4, 0, 6]]))
        y = tfe.define_private_variable(np.array([[0, -2, 3], [0, -5, 6]]), apply_scaling=False)

        z = tfe.equal_zero(x)
        w = tfe.equal_zero(y)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            result = sess.run(z.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[0, 0, 1], [0, 1, 0]]))
            result = sess.run(w.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[1, 0, 0], [1, 0, 0]]))

    def test_comparison(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(np.array([[1, -2, 0], [-4, 0, 6]]))
        y = tfe.define_private_variable(np.array([[0, -2, 3], [0, -5, 6]]))

        z1 = x > y
        z2 = x < y
        z3 = x >= y
        z4 = x <= y
        z5 = tfe.equal(x, y)
        z6 = x > 0
        z7 = x >= 0
        z8 = x < 0
        z9 = x <= 0

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[1, 0, 0], [0, 1, 0]]))

            result = sess.run(z2.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[0, 0, 1], [1, 0, 0]]))

            result = sess.run(z3.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[1, 1, 0], [0, 1, 1]]))

            result = sess.run(z4.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[0, 1, 1], [1, 0, 1]]))

            result = sess.run(z5.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[0, 1, 0], [0, 0, 1]]))

            result = sess.run(z6.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[1, 0, 0], [0, 0, 1]]))

            result = sess.run(z7.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[1, 0, 1], [0, 1, 1]]))

            result = sess.run(z8.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[0, 1, 0], [1, 0, 0]]))

            result = sess.run(z9.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[0, 1, 1], [1, 1, 0]]))


    def test_pow_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant([[1, 2, 3], [4, 5, 6]]))

        y = x ** 2
        z = x ** 3

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(y.reveal())
            np.testing.assert_allclose(
                result, np.array([[1, 4, 9], [16, 25, 36]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z.reveal())
            np.testing.assert_allclose(
                result, np.array([[1, 8, 27], [64, 125, 216]]), rtol=0.0, atol=0.01
            )

    def test_polynomial_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant([[1, 2, 3], [4, 5, 6]]))

        # Friendly version
        y = 1 + 1.2 * x + 3 * (x ** 2) + 0.5 * (x ** 3)
        # More optimized version: No truncation for multiplying integer coefficients (e.g., '3' in this example)
        z = tfe.polynomial(x, [1, 1.2, 3, 0.5])

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(y.reveal())
            np.testing.assert_allclose(
                result,
                np.array([[5.7, 19.4, 45.1], [85.8, 144.5, 224.2]]),
                rtol=0.0,
                atol=0.01,
            )

            result = sess.run(z.reveal())
            np.testing.assert_allclose(
                result,
                np.array([[5.7, 19.4, 45.1], [85.8, 144.5, 224.2]]),
                rtol=0.0,
                atol=0.01,
            )

    def test_polynomial_piecewise(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant([[-1, -0.5, -0.25], [0, 0.25, 2]]))

        # This is the approximation of the sigmoid function by using a piecewise function:
        # f(x) = (0 if x<-0.5), (x+0.5 if -0.5<=x<0.5), (1 if x>=0.5)
        z1 = tfe.polynomial_piecewise(
            x,
            (-0.5, 0.5),
            ((0,), (0.5, 1), (1,)),  # use tuple because list is not hashable
        )
        # Or, simply use the pre-defined sigmoid API which includes a different approximation
        z2 = tfe.sigmoid(x)
        z3 = tfe.relu(x)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[0, 0, 0.25], [0.5, 0.75, 1]]), rtol=0.0, atol=0.01
            )
            result = sess.run(z2.reveal())
            np.testing.assert_allclose(
                result,
                np.array([[0.33, 0.415, 0.4575], [0.5, 0.5425, 0.84]]),
                rtol=0.0,
                atol=0.01,
            )
            result = sess.run(z3.reveal())
            np.testing.assert_allclose(
                result,
                np.array([[0, 0, 0], [0, 0.25, 2]]),
                rtol=0.0,
                atol=0.01,
            )

    def test_reciprocal(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant([0.001, 0.25, 1, 2, 5, 10, 20]))

        z = tfe.reciprocal(x)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            # np.testing.assert_allclose(
                # result, np.array([1000, 4, 1, 0.5, 0.2, 0.1, 0.05]), rtol=0.0, atol=0.01
            # )

    def test_exp(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant([-4, -0.5, 0, 1, 2, 5]))

        z = tfe.exp(x)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal())
            # np.testing.assert_allclose(
                # result, np.array([1.83156389e-02, 6.06530660e-01, 1, 2.71828183e+00, 7.38905610e+00, 1.48413159e+02]), rtol=0.0, atol=0.01
            # )

    def test_transpose(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant([[1, 2, 3], [4, 5, 6]]))
        y = tfe.define_constant(np.array([[1, 2, 3], [4, 5, 6]]))

        z1 = x.transpose()
        z2 = tfe.transpose(y)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[1, 4], [2, 5], [3, 6]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z2)
            np.testing.assert_allclose(
                result, np.array([[1, 4], [2, 5], [3, 6]]), rtol=0.0, atol=0.01
            )

    def test_reduce_sum(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant([[1, 2, 3], [4, 5, 6]]))
        y = tfe.define_constant(np.array([[1, 2, 3], [4, 5, 6]]))

        z1 = x.reduce_sum(axis=1, keepdims=True)
        z2 = tfe.reduce_sum(y, axis=0, keepdims=False)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[6], [15]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z2)
            np.testing.assert_allclose(result, np.array([5, 7, 9]), rtol=0.0, atol=0.01)

    def test_concat(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x1 = tfe.define_private_variable(tf.constant([[1, 2], [4, 5]]))
        x2 = tfe.define_private_variable(tf.constant([[3], [6]]))
        y1 = tfe.define_constant(np.array([[1, 2, 3]]))
        y2 = tfe.define_constant(np.array([[4, 5, 6]]))

        z1 = tfe.concat([x1, x2], axis=1)
        z2 = tfe.concat([y1, y2], axis=0)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[1, 2, 3], [4, 5, 6]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z2)
            np.testing.assert_allclose(
                result, np.array([[1, 2, 3], [4, 5, 6]]), rtol=0.0, atol=0.01
            )

    def test_stack(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x1 = tfe.define_private_variable(tf.constant([[1, 2], [3, 4]]))
        x2 = tfe.define_private_variable(tf.constant([[5, 6], [7, 8]]))
        y1 = tfe.define_constant(np.array([1, 2, 3]))
        y2 = tfe.define_constant(np.array([4, 5, 6]))

        z1 = tfe.stack([x1, x2], axis=1)
        z2 = tfe.stack([y1, y2], axis=0)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z2)
            np.testing.assert_allclose(
                result, np.array([[1, 2, 3], [4, 5, 6]]), rtol=0.0, atol=0.01
            )

    def test_expand_dims(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant([[1, 2], [3, 4]]))
        y = tfe.define_constant(np.array([1, 2, 3]))

        z1 = tfe.expand_dims(x, axis=1)
        z2 = tfe.expand_dims(y, axis=1)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[[1, 2]], [[3, 4]]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z2)
            np.testing.assert_allclose(
                result, np.array([[1], [2], [3]]), rtol=0.0, atol=0.01
            )

    def test_squeeze(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant([[[1, 2]], [[3, 4]]]))
        y = tfe.define_constant(np.array([[1], [2], [3]]))

        z1 = tfe.squeeze(x)
        z2 = tfe.squeeze(y, axis=[1])

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[1, 2], [3, 4]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z2)
            np.testing.assert_allclose(
                result, np.array([1, 2, 3]), rtol=0.0, atol=0.01
            )

    def test_strided_slice(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant(
                [[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]]))

        z1 = tfe.strided_slice(x, [1, 0, 0], [2, 1, 3], [1, 1, 1])
        z2 = tfe.strided_slice(x, [1, 0, 0], [2, 2, 3], [1, 1, 1])
        z3 = tfe.strided_slice(x, [1, -1, 0], [2, -3, 3], [1, -1, 1])

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[[3, 3, 3]]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z2.reveal())
            np.testing.assert_allclose(
                result, np.array([[[3, 3, 3], [4, 4, 4]]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z3.reveal())
            np.testing.assert_allclose(
                result, np.array([[[4, 4, 4], [3, 3, 3]]]), rtol=0.0, atol=0.01
            )

    def test_gather(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant([[1, 2, 3], [4, 5, 6]]))
        y = tfe.define_constant(np.array([[1, 2, 3], [4, 5, 6]]))

        z1 = tfe.gather(x, [0, 2], axis=1)
        z2 = tfe.gather(y, [0, 2], axis=1)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[1, 3], [4, 6]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z2)
            np.testing.assert_allclose(
                result, np.array([[1, 3], [4, 6]]), rtol=0.0, atol=0.01
            )

    def test_split(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]))
        y = tfe.define_constant(np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]))

        w = tfe.split(x, 3, axis=1)
        z = tfe.split(y, 3, axis=1)
        assert len(w) == 3
        assert len(z) == 3

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(w[0].reveal())
            np.testing.assert_allclose(
                result, np.array([[1, 2], [7, 8]]), rtol=0.0, atol=0.01
            )
            result = sess.run(w[1].reveal())
            np.testing.assert_allclose(
                result, np.array([[3, 4], [9, 10]]), rtol=0.0, atol=0.01
            )
            result = sess.run(w[2].reveal())
            np.testing.assert_allclose(
                result, np.array([[5, 6], [11, 12]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z[0])
            np.testing.assert_allclose(
                result, np.array([[1, 2], [7, 8]]), rtol=0.0, atol=0.01
            )
            result = sess.run(z[1])
            np.testing.assert_allclose(
                result, np.array([[3, 4], [9, 10]]), rtol=0.0, atol=0.01
            )
            result = sess.run(z[2])
            np.testing.assert_allclose(
                result, np.array([[5, 6], [11, 12]]), rtol=0.0, atol=0.01
            )

    def test_tile(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.constant([[1, 2, 3], [4, 5, 6]]))
        y = tfe.define_constant(np.array([[1, 2, 3], [4, 5, 6]]))

        z1 = tfe.tile(x, [1, 2])
        z2 = tfe.tile(y, [1, 2])

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z2)
            np.testing.assert_allclose(
                result, np.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]]), rtol=0.0, atol=0.01
            )

    def test_im2col(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(tf.random.uniform([128, 3, 27, 27]))
        y = tfe.define_constant(np.random.uniform(size = [128, 3, 27, 27]))

        z1 = tfe.im2col(x, 5, 5, "SAME", 2)
        z2 = tfe.im2col(y, 5, 5, "VALID", 2)

        n_rows_same = math.ceil(27 / 2)
        n_cols_same = math.ceil(27 / 2)
        n_rows_valid = math.ceil((27 - 5 + 1) / 2)
        n_cols_valid = math.ceil((27 - 5 + 1) / 2)
        assert z1.shape == [5 * 5 * 3, n_rows_same * n_cols_same * 128]
        assert z2.shape == [5 * 5 * 3, n_rows_valid * n_cols_valid * 128]


    def test_simple_lr_model(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        # define inputs
        x_raw = tf.random.uniform(minval=-0.5, maxval=0.5, shape=[99, 10], seed=1000)
        x = tfe.define_private_variable(x_raw, name="x")
        y_raw = tf.cast(
            tf.reduce_mean(x_raw, axis=1, keepdims=True) > 0, dtype=tf.float32
        )
        y = tfe.define_private_variable(y_raw, name="y")
        w = tfe.define_private_variable(
            tf.random_uniform([10, 1], -0.01, 0.01, seed=100), name="w"
        )
        b = tfe.define_private_variable(tf.zeros([1]), name="b")
        learning_rate = 0.01

        with tf.name_scope("forward"):
            out = tfe.matmul(x, w) + b
            y_hat = tfe.sigmoid(out)

        with tf.name_scope("loss-grad"):
            dy = y_hat - y
        batch_size = x.shape.as_list()[0]
        with tf.name_scope("backward"):
            dw = tfe.matmul(tfe.transpose(x), dy) / batch_size
            db = tfe.reduce_sum(dy, axis=0) / batch_size
            upd1 = dw * learning_rate
            upd2 = db * learning_rate
            assign_ops = [tfe.assign(w, w - upd1), tfe.assign(b, b - upd2)]

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            sess.run(assign_ops)

    def test_mul_trunc2_private_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        @tfe.local_computation("input-provider")
        def provide_input():
            # normal TensorFlow operations can be run locally
            # as part of defining a private input, in this
            # case on the machine of the input provider
            return tf.ones(shape=(2, 2)) * 1.3

        # define inputs
        x = tfe.define_private_variable(tf.ones(shape=(2, 2)) * 2)
        y = provide_input()

        # define computation
        z = tfe.mul_trunc2(x, y)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal(), tag="mul_trunc2")
            np.testing.assert_allclose(
                result, np.array([[2.6, 2.6], [2.6, 2.6]]), rtol=0.0, atol=0.01
            )

    def test_write_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        @tfe.local_computation("input-provider")
        def provide_input():
            # normal TensorFlow operations can be run locally
            # as part of defining a private input, in this
            # case on the machine of the input provider
            return tf.ones(shape=(2, 2)) * 1.3

        # define inputs
        x = provide_input()

        _, tmp_filename = tempfile.mkstemp()
        write_op = x.write(tmp_filename)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            sess.run(write_op)

        os.remove(tmp_filename)

    def test_read_private(self):

        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        @tfe.local_computation("input-provider")
        def provide_input():
            return tf.reshape(tf.range(0, 8), [4, 2])

        # define inputs
        x = provide_input()

        _, tmp_filename = tempfile.mkstemp()
        write_op = x.write(tmp_filename)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            sess.run(write_op)

        x = tfe.read(tmp_filename, batch_size=5, n_columns=2)
        with tfe.Session() as sess:
            result = sess.run(x.reveal())
            np.testing.assert_allclose(
                result,
                np.array(list(range(0, 8)) + [0, 1]).reshape([5, 2]),
                rtol=0.0,
                atol=0.01,
            )

        os.remove(tmp_filename)

    @unittest.skip
    def test_iterate_private(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        @tfe.local_computation("input-provider")
        def provide_input():
            return tf.reshape(tf.range(0, 8), [4, 2])

        # define inputs
        x = provide_input()

        _, tmp_filename = tempfile.mkstemp()
        write_op = x.write(tmp_filename)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            sess.run(write_op)

        x = tfe.read(tmp_filename, batch_size=5, n_columns=2)
        y = tfe.iterate(x, batch_size=3, repeat=True, shuffle=False)
        z = tfe.iterate(x, batch_size=3, repeat=True, shuffle=True)
        with tfe.Session() as sess:
            sess.run(tfe.global_variables_initializer())
            # TODO: fix this test
            print(sess.run(x.reveal()))
            print(sess.run(y.reveal()))
            print(sess.run(y.reveal()))
            print(sess.run(x.reveal()))
            print(sess.run(z.reveal()))

        os.remove(tmp_filename)

    def test_conv2d(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        # input
        batch_size, channels_in, channels_out = 32, 3, 64
        img_height, img_width = 28, 28
        input_shape = (batch_size, channels_in, img_height, img_width)
        input_conv = np.random.normal(size=input_shape).astype(np.float32)

        # filters
        h_filter, w_filter, strides = 2, 2, 2
        filter_shape = (h_filter, w_filter, channels_in, channels_out)
        filter_values = np.random.normal(size=filter_shape)

        conv_input = prot.define_private_variable(input_conv)
        private_filter = prot.define_private_variable(filter_values)
        conv_output_tfe1 = tfe.conv2d(conv_input, filter_values, strides=2, padding="SAME")
        conv_output_tfe2 = tfe.conv2d(conv_input, private_filter, strides=2, padding="SAME")

        with tfe.Session() as sess:

            sess.run(tf.global_variables_initializer())
            # outputs
            output_tfe1 = sess.run(conv_output_tfe1.reveal())
            output_tfe2 = sess.run(conv_output_tfe2.reveal())

        # reset graph
        tf.reset_default_graph()

        # convolution tensorflow
        with tf.Session() as sess:
            # conv input
            x = tf.Variable(input_conv, dtype=tf.float32)
            x_nhwc = tf.transpose(x, (0, 2, 3, 1))

            # convolution Tensorflow
            filters_tf = tf.Variable(filter_values, dtype=tf.float32)

            conv_out_tf = tf.nn.conv2d(
                x_nhwc, filters_tf, strides=[1, strides, strides, 1], padding="SAME",
            )

            sess.run(tf.global_variables_initializer())
            output_tensorflow = sess.run(conv_out_tf).transpose(0, 3, 1, 2)

        np.testing.assert_allclose(output_tfe1, output_tensorflow, atol=0.01)
        np.testing.assert_allclose(output_tfe2, output_tensorflow, atol=0.01)

    def test_maximum(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(np.array([[1, -2, 0], [-4, 0, 6]]))
        y = tfe.define_private_variable(np.array([[0, -2, 3], [0, -5, 6]]))

        z = tfe.maximum(x, y)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            result = sess.run(z.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[1, -2, 3], [0, 0, 6]]))

    def test_reduce_max(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(np.array([[1, -2, 0], [-4, 0, 6]]))

        z1 = tfe.reduce_max(x, axis=0)
        z2 = tfe.reduce_max(x, axis=0, keepdims=True)
        z3 = tfe.reduce_max(x, axis=1)
        z4 = tfe.reduce_max(x, axis=1, keepdims=True)
        z5 = tfe.reduce_max(x)
        z6 = tfe.reduce_max(x, keepdims=True)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            result = sess.run(z1.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([1, 0, 6]))

            result = sess.run(z2.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[1, 0, 6]]))

            result = sess.run(z3.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([1, 6]))

            result = sess.run(z4.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[1], [6]]))

            result = sess.run(z5.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array(6))

            result = sess.run(z6.reveal())
            np.testing.assert_array_equal(result.astype(int), np.array([[6]]))

    def test_maxpool2d(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(np.array([[1, 2, 3, 4],
                                                  [5, 6, 7, 8],
                                                  [9, 10, 11, 12]]))
        # Add the `batch` and `channels` dimensions
        x = tfe.expand_dims(x, axis=0)
        x = tfe.expand_dims(x, axis=0)

        z1 = tfe.maxpool2d(x, pool_size=(2, 2), strides=(2, 2), padding="VALID")
        z2 = tfe.maxpool2d(x, pool_size=(2, 2), strides=(2, 2), padding="SAME")

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(result, np.array([[[[6, 8]]]]), rtol=0.0, atol=0.01)

            result = sess.run(z2.reveal())
            np.testing.assert_allclose(result, np.array([[[[6, 8], [10, 12]]]]), rtol=0.0, atol=0.01)

    def test_avgpool2d(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(np.array([[1, 2, 3, 4],
                                                  [5, 6, 7, 8],
                                                  [9, 10, 11, 12]]))
        # Add the `batch` and `channels` dimensions
        x = tfe.expand_dims(x, axis=0)
        x = tfe.expand_dims(x, axis=0)

        z1 = tfe.avgpool2d(x, pool_size=(2, 2), strides=(2, 2), padding="VALID")
        z2 = tfe.avgpool2d(x, pool_size=(2, 2), strides=(2, 2), padding="SAME")

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(result, np.array([[[[3.5, 5.5]]]]), rtol=0.0, atol=0.01)

            result = sess.run(z2.reveal())
            np.testing.assert_allclose(result, np.array([[[[3.5, 5.5], [4.75, 5.75]]]]), rtol=0.0, atol=0.01)




def print_banner(title):
    title_length = len(title)
    banner_length = title_length + 2 * 10
    banner_top = "+" + ("-" * (banner_length - 2)) + "+"
    banner_middle = "|" + " " * 9 + title + " " * 9 + "|"

    print()
    print(banner_top)
    print(banner_middle)
    print(banner_top)


if __name__ == "__main__":
    """
    Run these tests with:
    python aby3_test.py
    """
    unittest.main()
