import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


class TestBatchToSpaceND(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(4224)

    def test4DNoCrops(self):
        backing = [[[[1], [3]], [[9], [11]]],
                   [[[2], [4]], [[10], [12]]],
                   [[[5], [7]], [[13], [15]]],
                   [[[6], [8]], [[14], [16]]]]
        t = tf.constant(backing)
        block_shape = [2, 2]
        crops = [[0, 0], [0, 0]]
        self._generic_private_test(t, block_shape, crops)

    def test4DSingleCrop(self):
        backing = [[[[0], [1], [3]]], [[[0], [9], [11]]],
                   [[[0], [2], [4]]], [[[0], [10], [12]]],
                   [[[0], [5], [7]]], [[[0], [13], [15]]],
                   [[[0], [6], [8]]], [[[0], [14], [16]]]]
        t = tf.constant(backing)
        block_shape = [2, 2]
        crops = [[0, 0], [2, 0]]
        self._generic_private_test(t, block_shape, crops)

    def test3DNoCrops(self):
        t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        crops = [[0, 0]]
        self._generic_private_test(t, block_shape, crops)

    def test3DMirrorCrops(self):
        t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        crops = [[2, 2]]
        self._generic_private_test(t, block_shape, crops)

    def test3DUnevenCrops(self):
        t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        crops = [[2, 0]]
        self._generic_private_test(t, block_shape, crops)

    def test3DBlockShape(self):
        t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [8]
        crops = [[0, 0]]
        self._generic_private_test(t, block_shape, crops)

    def testPublic(self):
        t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        crops = [[2, 2]]
        self._generic_public_test(t, block_shape, crops)

    def testMasked(self):
        t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        crops = [[2, 2]]
        self._generic_masked_test(t, block_shape, crops)

    @staticmethod
    def _generic_public_test(t, block_shape, crops):
        with tf.Session() as sess:
            out = tf.batch_to_space_nd(t, block_shape=block_shape, crops=crops)
            actual = sess.run(out)

        with tfe.protocol.Pond() as prot:
            b = prot.define_public_variable(t)
            out = prot.batch_to_space_nd(b, block_shape=block_shape, crops=crops)
            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out)

        np.testing.assert_array_almost_equal(final, actual, decimal=3)

    @staticmethod
    def _generic_private_test(t, block_shape, crops):
        with tf.Session() as sess:
            out = tf.batch_to_space_nd(t, block_shape=block_shape, crops=crops)
            actual = sess.run(out)

        with tfe.protocol.Pond() as prot:
            b = prot.define_private_variable(t)
            out = prot.batch_to_space_nd(b, block_shape=block_shape, crops=crops)
            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out.reveal())

        np.testing.assert_array_almost_equal(final, actual, decimal=3)

    @staticmethod
    def _generic_masked_test(t, block_shape, crops):
        with tf.Session() as sess:
            out = tf.batch_to_space_nd(t, block_shape=block_shape, crops=crops)
            actual = sess.run(out)

        with tfe.protocol.Pond() as prot:
            b = prot.mask(prot.define_private_variable(t))
            out = prot.batch_to_space_nd(b, block_shape=block_shape, crops=crops)
            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out.reveal())

        np.testing.assert_array_almost_equal(final, actual, decimal=3)


class TestSpaceToBatchND(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(4224)

    def test4DNoCrops(self):
        backing = [[[[1], [3]], [[9], [11]]],
                   [[[2], [4]], [[10], [12]]],
                   [[[5], [7]], [[13], [15]]],
                   [[[6], [8]], [[14], [16]]]]
        t = tf.constant(backing)
        block_shape = [2, 2]
        paddings = [[0, 0], [0, 0]]
        self._generic_private_test(t, block_shape, paddings)

    def test4DSingleCrop(self):
        backing = [[[[1], [2], [3], [4]],
                    [[5], [6], [7], [8]]],
                   [[[9], [10], [11], [12]],
                    [[13], [14], [15], [16]]]]
        t = tf.constant(backing)
        block_shape = [2, 2]
        paddings = [[0, 0], [2, 0]]
        self._generic_private_test(t, block_shape, paddings)

    def test3DNoCrops(self):
        t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        paddings = [[0, 0]]
        self._generic_private_test(t, block_shape, paddings)

    def test3DMirrorCrops(self):
        t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        paddings = [[2, 2]]
        self._generic_private_test(t, block_shape, paddings)

    def test3DUnevenCrops(self):
        t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [2]
        paddings = [[2, 0]]
        self._generic_private_test(t, block_shape, paddings)

    def test3DBlockShape(self):
        t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [5]
        paddings = [[0, 0]]
        self._generic_private_test(t, block_shape, paddings)

    def testPublic(self):
        t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        paddings = [[2, 2]]
        self._generic_public_test(t, block_shape, paddings)

    def testMasked(self):
        t = tf.random_uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        paddings = [[2, 2]]
        self._generic_masked_test(t, block_shape, paddings)

    @staticmethod
    def _generic_public_test(t, block_shape, paddings):
        with tf.Session() as sess:
            out = tf.space_to_batch_nd(t, block_shape=block_shape, paddings=paddings)
            actual = sess.run(out)

        with tfe.protocol.Pond() as prot:
            b = prot.define_public_variable(t)
            out = prot.space_to_batch_nd(b, block_shape=block_shape, paddings=paddings)
            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out)

        np.testing.assert_array_almost_equal(final, actual, decimal=3)

    @staticmethod
    def _generic_private_test(t, block_shape, paddings):
        with tf.Session() as sess:
            out = tf.space_to_batch_nd(t, block_shape=block_shape, paddings=paddings)
            actual = sess.run(out)

        with tfe.protocol.Pond() as prot:
            b = prot.define_private_variable(t)
            out = prot.space_to_batch_nd(b, block_shape=block_shape, paddings=paddings)
            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out.reveal())

        np.testing.assert_array_almost_equal(final, actual, decimal=3)

    @staticmethod
    def _generic_masked_test(t, block_shape, paddings):
        with tf.Session() as sess:
            out = tf.space_to_batch_nd(t, block_shape=block_shape, paddings=paddings)
            actual = sess.run(out)

        with tfe.protocol.Pond() as prot:
            b = prot.mask(prot.define_private_variable(t))
            out = prot.space_to_batch_nd(b, block_shape=block_shape, paddings=paddings)
            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())
                final = sess.run(out.reveal())

        np.testing.assert_array_almost_equal(final, actual, decimal=3)


if __name__ == "__main__":
    unittest.main()
