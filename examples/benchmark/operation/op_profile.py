# pylint: disable=all
# pylint: disable=missing-docstring
import argparse
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.performance import Performance
from tf_encrypted.protocol import ABY3  # noqa:F403,F401
from tf_encrypted.protocol import Pond  # noqa:F403,F401
from tf_encrypted.protocol import SecureNN  # noqa:F403,F401


class TestOPProfile(unittest.TestCase):
    def test_compare_performance(self):
        n = 2**10
        x = tf.range(n)
        x1 = tf.random.shuffle(x)
        x2 = tf.random.shuffle(x)
        actual = x1 > x2

        private_x1 = tfe.define_private_input("server0", lambda: x1)
        private_x2 = tfe.define_private_input("server0", lambda: x2)

        @tfe.function
        def test_compare(input_x1, input_x2):
            y = input_x1 > input_x2
            return y.reveal().to_native()

        Performance.time_log("1st run")
        result = test_compare(private_x1, private_x2)
        np.testing.assert_allclose(result, actual, rtol=0.0, atol=0.01)
        Performance.time_log("1st run")

        Performance.time_log("2nd run")
        test_compare(private_x1, private_x2)
        Performance.time_log("2nd run")

    def test_sort_performance(self):
        n = 2**10
        x = tf.range(n)
        x = tf.random.shuffle(x)
        actual = np.arange(n)
        private_x = tfe.define_private_input("server0", lambda: x)

        @tfe.function
        def test_sort(input):
            y = tfe.sort(input, axis=0, acc=True)
            return y.reveal().to_native()

        Performance.time_log("1st run")
        result = test_sort(private_x)
        np.testing.assert_allclose(result, actual, rtol=0.0, atol=0.01)
        Performance.time_log("1st run")

        Performance.time_log("2nd run")
        test_sort(private_x)
        Performance.time_log("2nd run")

    def test_max_performance_type1(self):
        n = 2**10
        x = tf.range(n)
        x = tf.random.shuffle(x)
        actual = tf.reduce_max(x, axis=0)
        private_x = tfe.define_private_input("server0", lambda: x)

        @tfe.function
        def test_max1(input):
            y = tfe.reduce_max(input, axis=0, method="network")
            return y.reveal().to_native()

        Performance.time_log("1st run")
        result = test_max1(private_x)
        np.testing.assert_allclose(result, actual, rtol=0.0, atol=0.01)
        Performance.time_log("1st run")

        Performance.time_log("2nd run")
        test_max1(private_x)
        Performance.time_log("2nd run")

    def test_max_performance_type2(self):
        n = 2**10 * 4
        x = tf.range(n)
        x = tf.reshape(tf.random.shuffle(x), [2**10, 4])
        actual = tf.reduce_max(x, axis=1)
        private_x = tfe.define_private_input("server0", lambda: x)

        @tfe.function
        def test_max2(input):
            y = tfe.reduce_max(input, axis=1)
            return y.reveal().to_native()

        Performance.time_log("1st run")
        result = test_max2(private_x)
        np.testing.assert_allclose(result, actual, rtol=0.0, atol=0.01)
        Performance.time_log("1st run")

        Performance.time_log("2nd run")
        test_max2(private_x)
        Performance.time_log("2nd run")

    def test_log_performance(self):
        n = 2**10
        x = tf.random.uniform(shape=[n], minval=0, maxval=1)
        private_x = tfe.define_private_input("server0", lambda: x)

        @tfe.function
        def test_log(input):
            y = tfe.log(input)
            return y.reveal().to_native()

        Performance.time_log("1st run")
        test_log(private_x)
        Performance.time_log("1st run")

        Performance.time_log("2nd run")
        test_log(private_x)
        Performance.time_log("2nd run")

    def test_matmul_performance(self):
        n = 1024
        repeats = 10
        x = tf.random.uniform([n, n])
        y = tf.random.uniform([n, n])
        actual = tf.matmul(x, y)
        private_x = tfe.define_private_input("server0", lambda: x)
        private_y = tfe.define_private_input("server0", lambda: y)

        @tfe.function
        def test_matmul(input_x, input_y):
            z = tfe.matmul(input_x, input_y)
            return z.reveal().to_native()

        Performance.time_log("1st run")
        result = test_matmul(private_x, private_y)
        np.testing.assert_allclose(result, actual, rtol=0.0, atol=0.01)
        Performance.time_log("1st run")

        Performance.time_log("run " + str(repeats) + " rounds")
        for i in range(repeats):
            test_matmul(private_x, private_y)
        Performance.time_log("run " + str(repeats) + " rounds")

    def test_cwise_performance(self):
        repeats = 10
        n = 1024
        x = tf.random.uniform([n, n])
        y = tf.random.uniform([n, n])
        private_x = tfe.define_private_input("server0", lambda: x)
        private_y = tfe.define_private_input("server0", lambda: y)

        @tfe.function
        def test_add(input_x, input_y):
            z = input_x + input_y
            return z.reveal().to_native()

        @tfe.function
        def test_sub(input_x, input_y):
            z = input_x - input_y
            return z.reveal().to_native()

        @tfe.function
        def test_mul(input_x, input_y):
            z = input_x * input_y
            return z.reveal().to_native()

        @tfe.function
        def test_add_constant(input_x):
            z = input_x + 1
            return z.reveal().to_native()

        @tfe.function
        def test_mul_constant(input_x):
            z = input_x * 2
            return z.reveal().to_native()

        Performance.time_log("Add 1st run")
        result = test_add(private_x, private_y)
        np.testing.assert_allclose(result, x + y, rtol=0.0, atol=0.01)
        Performance.time_log("Add 1st run")

        Performance.time_log("Add run " + str(repeats) + " rounds")
        for i in range(repeats):
            test_add(private_x, private_y)
        Performance.time_log("Add run " + str(repeats) + " rounds")

        Performance.time_log("Sub 1st run")
        result = test_sub(private_x, private_y)
        np.testing.assert_allclose(result, x - y, rtol=0.0, atol=0.01)
        Performance.time_log("Sub 1st run")

        Performance.time_log("Sub run " + str(repeats) + " rounds")
        for i in range(repeats):
            test_sub(private_x, private_y)
        Performance.time_log("Sub run " + str(repeats) + " rounds")

        Performance.time_log("Mul 1st run")
        result = test_mul(private_x, private_y)
        np.testing.assert_allclose(result, tf.multiply(x, y), rtol=0.0, atol=0.01)
        Performance.time_log("Mul 1st run")

        Performance.time_log("Mul run " + str(repeats) + " rounds")
        for i in range(repeats):
            test_mul(private_x, private_y)
        Performance.time_log("Mul run " + str(repeats) + " rounds")

        Performance.time_log("Add const 1st run")
        result = test_add_constant(private_x)
        np.testing.assert_allclose(result, x + 1, rtol=0.0, atol=0.01)
        Performance.time_log("Add const 1st run")

        Performance.time_log("Add const run " + str(repeats) + " rounds")
        for i in range(repeats):
            test_add_constant(private_x)
        Performance.time_log("Add const run " + str(repeats) + " rounds")

        Performance.time_log("Mul const 1st run")
        result = test_mul_constant(private_x)
        np.testing.assert_allclose(result, x * 2, rtol=0.0, atol=0.01)
        Performance.time_log("Mul const 1st run")

        Performance.time_log("Mul const run " + str(repeats) + " rounds")
        for i in range(repeats):
            test_mul_constant(private_x)
        Performance.time_log("Mul const run " + str(repeats) + " rounds")


if __name__ == "__main__":
    """
    Run these tests with:
    python op_profile.py
    """

    parser = argparse.ArgumentParser(
        description="Run a TF Encrypted operation benchmark"
    )
    parser.add_argument(
        "test_case",
        metavar="TEST CASE",
        type=str,
        help="which test to run",
    )
    parser.add_argument(
        "--protocol",
        metavar="PROTOCOL",
        type=str,
        default="ABY3",
        help="MPC protocol TF Encrypted used",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=str,
        default="./config.json",
        help="path to configuration file",
    )
    parser.add_argument(
        "--precision",
        choices=["l", "h", "low", "high"],
        type=str,
        default="l",
        help="use 64 or 128 bits for computation",
    )
    args = parser.parse_args()

    # set tfe config
    if args.config != "local":
        # config file was specified
        config_file = args.config
        config = tfe.RemoteConfig.load(config_file)
        config.connect_servers()
        tfe.set_config(config)
    else:
        # Always best practice to preset all players to avoid invalid device errors
        config = tfe.LocalConfig(player_names=["server0", "server1", "server2"])
        tfe.set_config(config)

    # set tfe protocol
    tfe.set_protocol(globals()[args.protocol](fixedpoint_config=args.precision))

    test = args.test_case
    singletest = unittest.TestSuite()
    singletest.addTest(TestOPProfile(test))
    unittest.TextTestRunner().run(singletest)
