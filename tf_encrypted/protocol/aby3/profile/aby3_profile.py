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

import tf_encrypted as tfe
from tf_encrypted.performance import Performance
from tf_encrypted.protocol.aby3 import ABY3
from tf_encrypted.protocol.aby3 import ShareType


class TestABY3Profile(unittest.TestCase):
    def test_sort_performance(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        n = 2 ** 10
        x = tf.range(n)
        x = tf.random.shuffle(x)
        private_x = tfe.define_private_variable(x)

        Performance.time_log("Graph building")
        y0 = tfe.sort(private_x, axis=0, acc=True)
        Performance.time_log("Graph building")

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            Performance.time_log("1st run")
            x_, result = sess.run([x, y0.reveal()])
            np.testing.assert_allclose(result, np.arange(n), rtol=0.0, atol=0.01)
            Performance.time_log("1st run")
            print("1st run x: ", x_[:10], "...")
            Performance.time_log("2nd run")
            x_, result = sess.run([x, y0.reveal()])
            Performance.time_log("2nd run")

    def test_max_performance_type1(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        n = 2 ** 10
        x = tf.range(n)
        x = tf.random.shuffle(x)
        private_x = tfe.define_private_variable(x)

        Performance.time_log("Graph building")
        y0 = tfe.reduce_max(private_x, axis=0, method="network")
        Performance.time_log("Graph building")

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            Performance.time_log("1st run")
            x_, result = sess.run([x, y0.reveal()])
            Performance.time_log("1st run")
            Performance.time_log("2nd run")
            x_, result = sess.run([x, y0.reveal()])
            Performance.time_log("2nd run")

    def test_max_performance_type2(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        n = 2 ** 10 * 4
        x = tf.range(n)
        x = tf.reshape(tf.random.shuffle(x), [2 ** 10, 4])
        private_x = tfe.define_private_variable(x)

        Performance.time_log("Graph building")
        y0 = tfe.reduce_max(private_x, axis=1)
        Performance.time_log("Graph building")

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            Performance.time_log("1st run")
            x_, result = sess.run([x, y0.reveal()])
            Performance.time_log("1st run")
            Performance.time_log("2nd run")
            x_, result = sess.run([x, y0.reveal()])
            Performance.time_log("2nd run")


if __name__ == "__main__":
    """
    Run these tests with:
    python aby3_profile.py
    """

    if len(sys.argv) < 3:
        raise RuntimeError("Expect at least 3 arguments")

    # config file was specified
    config_file = sys.argv[1]
    config = tfe.RemoteConfig.load(config_file)
    tfe.set_config(config)

    test = sys.argv[2]
    singletest = unittest.TestSuite()
    singletest.addTest(TestABY3Profile(test))
    unittest.TextTestRunner().run(singletest)
