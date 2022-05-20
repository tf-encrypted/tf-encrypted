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
from .performance import Performance


class TestABY3Profile(unittest.TestCase):


    def test_sort_performance(self):
        tf.reset_default_graph()

        prot = ABY3()
        tfe.set_protocol(prot)

        n = 2**20
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
            np.testing.assert_allclose(
                result, np.arange(n), rtol=0.0, atol=0.01
            )
            Performance.time_log("1st run")
            print("1st run x: ", x_[:10], "...")
            Performance.time_log("2nd run")
            x_, result = sess.run([x, y0.reveal()])
            Performance.time_log("2nd run")



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
