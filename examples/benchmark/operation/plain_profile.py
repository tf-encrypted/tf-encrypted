# pylint: disable=all
# pylint: disable=missing-docstring

import sys
import unittest

import tensorflow as tf

from tf_encrypted.performance import Performance


class TestPlainProfile(unittest.TestCase):
    def test_compare_performance(self):
        n = 2**10
        x = tf.range(n)
        x1 = tf.random.shuffle(x)
        x2 = tf.random.shuffle(x)

        @tf.function
        def test_compare(input_x1, input_x2):
            y = input_x1 > input_x2
            return y

        Performance.time_log("1st run")
        test_compare(x1, x2)
        Performance.time_log("1st run")

        Performance.time_log("2nd run")
        test_compare(x1, x2)
        Performance.time_log("2nd run")

    def test_sort_performance(self):
        n = 2**10
        x = tf.range(n)
        x = tf.random.shuffle(x)

        @tf.function
        def test_sort(input):
            y, _ = tf.math.top_k(x, k=n, sorted=True)
            return y

        Performance.time_log("1st run")
        test_sort(x)
        Performance.time_log("1st run")

        Performance.time_log("2nd run")
        test_sort(x)
        Performance.time_log("2nd run")

    def test_max_performance_type1(self):
        n = 2**10
        x = tf.range(n)
        x = tf.random.shuffle(x)

        @tf.function
        def test_max1(input):
            y = tf.reduce_max(input, axis=0)
            return y

        Performance.time_log("1st run")
        test_max1(x)
        Performance.time_log("1st run")

        Performance.time_log("2nd run")
        test_max1(x)
        Performance.time_log("2nd run")

    def test_max_performance_type2(self):
        n = 2**10 * 4
        x = tf.range(n)
        x = tf.reshape(tf.random.shuffle(x), [2**10, 4])

        @tf.function
        def test_max2(input):
            y = tf.reduce_max(input, axis=1)
            return y

        Performance.time_log("1st run")
        test_max2(x)
        Performance.time_log("1st run")

        Performance.time_log("2nd run")
        test_max2(x)
        Performance.time_log("2nd run")

    def test_log_performance(self):
        n = 2**10
        x = tf.random.uniform(shape=[n], minval=0, maxval=1)

        @tf.function
        def test_log(input):
            y = tf.math.log(input)
            return y

        Performance.time_log("1st run")
        test_log(x)
        Performance.time_log("1st run")

        Performance.time_log("2nd run")
        test_log(x)
        Performance.time_log("2nd run")

    def test_matmul_performance(self):
        n = 1024
        repeats = 10
        x = tf.random.uniform([n, n])
        y = tf.random.uniform([n, n])

        @tf.function
        def test_matmul(input_x, input_y):
            z = tf.matmul(input_x, input_y)
            return z

        Performance.time_log("1st run")
        test_matmul(x, y)
        Performance.time_log("1st run")

        Performance.time_log("run " + str(repeats) + " rounds")
        for i in range(repeats):
            test_matmul(x, y)
        Performance.time_log("run " + str(repeats) + " rounds")

    def test_cwise_performance(self):
        repeats = 10
        n = 1024
        x = tf.random.uniform([n, n])
        y = tf.random.uniform([n, n])

        @tf.function
        def test_add(input_x, input_y):
            z = input_x + input_y
            return z

        @tf.function
        def test_sub(input_x, input_y):
            z = input_x - input_y
            return z

        @tf.function
        def test_mul(input_x, input_y):
            z = input_x * input_y
            return z

        @tf.function
        def test_add_constant(input_x):
            z = input_x + 1
            return z

        @tf.function
        def test_mul_constant(input_x):
            z = input_x * 2
            return z

        Performance.time_log("Add 1st run")
        test_add(x, y)
        Performance.time_log("Add 1st run")

        Performance.time_log("Add run " + str(repeats) + " rounds")
        for i in range(repeats):
            test_add(x, y)
        Performance.time_log("Add run " + str(repeats) + " rounds")

        Performance.time_log("Sub 1st run")
        test_sub(x, y)
        Performance.time_log("Sub 1st run")

        Performance.time_log("Sub run " + str(repeats) + " rounds")
        for i in range(repeats):
            test_sub(x, y)
        Performance.time_log("Sub run " + str(repeats) + " rounds")

        Performance.time_log("Mul 1st run")
        test_mul(x, y)
        Performance.time_log("Mul 1st run")

        Performance.time_log("Mul run " + str(repeats) + " rounds")
        for i in range(repeats):
            test_mul(x, y)
        Performance.time_log("Mul run " + str(repeats) + " rounds")

        Performance.time_log("Add const 1st run")
        test_add_constant(x)
        Performance.time_log("Add const 1st run")

        Performance.time_log("Add const run " + str(repeats) + " rounds")
        for i in range(repeats):
            test_add_constant(x)
        Performance.time_log("Add const run " + str(repeats) + " rounds")

        Performance.time_log("Mul const 1st run")
        test_mul_constant(x)
        Performance.time_log("Mul const 1st run")

        Performance.time_log("Mul const run " + str(repeats) + " rounds")
        for i in range(repeats):
            test_mul_constant(x)
        Performance.time_log("Mul const run " + str(repeats) + " rounds")


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
