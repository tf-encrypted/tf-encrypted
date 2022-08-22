# pylint: disable=all
# pylint: disable=missing-docstring
# flake8: noqa

import math
import os
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.performance import Performance


class TestPlainProfile(unittest.TestCase):
    def test_sort_performance(self):
        tf.reset_default_graph()

        Performance.time_log("Graph building")
        n = 2 ** 10
        x = tf.range(n)
        x = tf.random.shuffle(x)
        y, _ = tf.math.top_k(x, k=n, sorted=True)
        Performance.time_log("Graph building")

        with tf.Session() as sess:
            # initialize variables
            sess.run(tf.global_variables_initializer())
            # reveal result
            Performance.time_log("1st run")
            x_, result = sess.run([x, y])
            np.testing.assert_allclose(result, np.flip(np.arange(n)), rtol=0.0, atol=0.01)
            Performance.time_log("1st run")
            Performance.time_log("2nd run")
            x_, result = sess.run([x, y])
            Performance.time_log("2nd run")

    def test_max_performance_type1(self):
        tf.reset_default_graph()

        Performance.time_log("Graph building")
        n = 2 ** 10
        x = tf.range(n)
        x = tf.random.shuffle(x)
        y = tf.reduce_max(x, axis=0)
        Performance.time_log("Graph building")

        with tf.Session() as sess:
            # initialize variables
            sess.run(tf.global_variables_initializer())
            # reveal result
            Performance.time_log("1st run")
            x_, result = sess.run([x, y])
            Performance.time_log("1st run")
            Performance.time_log("2nd run")
            x_, result = sess.run([x, y])
            Performance.time_log("2nd run")

    def test_max_performance_type2(self):
        tf.reset_default_graph()

        Performance.time_log("Graph building")
        n = 2 ** 10 * 4
        x = tf.range(n)
        x = tf.reshape(tf.random.shuffle(x), [2 ** 10, 4])
        y = tf.reduce_max(x, axis=1)
        Performance.time_log("Graph building")

        with tf.Session() as sess:
            # initialize variables
            sess.run(tf.global_variables_initializer())
            # reveal result
            Performance.time_log("1st run")
            x_, result = sess.run([x, y])
            Performance.time_log("1st run")
            Performance.time_log("2nd run")
            x_, result = sess.run([x, y])
            Performance.time_log("2nd run")


    def test_matmul_performance(self):
        tf.reset_default_graph()

        n = 1000000
        x = tf.reshape(tf.range(n), [1, n])
        y = tf.reshape(tf.range(n), [n, 1])

        Performance.time_log("Graph building")
        z = tf.matmul(x, y)
        Performance.time_log("Graph building")

        with tf.Session() as sess:
            # initialize variables
            sess.run(tf.global_variables_initializer())
            # reveal result
            sess.run([z])
            Performance.time_log("Matmul")
            for i in range(1):
                result = sess.run([z])
            Performance.time_log("Matmul")


    def test_cwise_performance(self):
        tf.reset_default_graph()

        repeats = 10
        n = 100000
        x = tf.reshape(tf.range(n), [1, n])
        y = tf.reshape(tf.range(n), [1, n])

        Performance.time_log("Graph building")
        z0 = x + y
        z1 = x - y
        z2 = x * y
        z3 = x + 1
        z4 = x * 2
        Performance.time_log("Graph building")

        with tf.Session() as sess:
            # initialize variables
            sess.run(tf.global_variables_initializer())
            # reveal result
            sess.run([z0, z1, z2])
            Performance.time_log("Add")
            for i in range(repeats):
                result = sess.run([z0])
            Performance.time_log("Add")
            Performance.time_log("Sub")
            for i in range(repeats):
                result = sess.run([z1])
            Performance.time_log("Sub")
            Performance.time_log("Mul")
            for i in range(repeats):
                result = sess.run([z2])
            Performance.time_log("Mul")
            Performance.time_log("Add constant")
            for i in range(repeats):
                result = sess.run([z3])
            Performance.time_log("Add constant")
            Performance.time_log("Mul constant")
            for i in range(repeats):
                result = sess.run([z4])
            Performance.time_log("Mul constant")

if __name__ == "__main__":
    """
    Run these tests with:
    python plain_profile.py
    """

    if len(sys.argv) < 2:
        raise RuntimeError("Expect at least 2 arguments")

    test = sys.argv[1]
    singletest = unittest.TestSuite()
    singletest.addTest(TestPlainProfile(test))
    unittest.TextTestRunner().run(singletest)
