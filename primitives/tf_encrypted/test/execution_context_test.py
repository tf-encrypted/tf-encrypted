# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from tf_encrypted.test import tf_execution_context


class TestExecutionContext(parameterized.TestCase):
    @parameterized.parameters({"run_eagerly": True}, {"run_eagerly": False})
    def test_tf_execution_mode(self, run_eagerly):
        context = tf_execution_context(run_eagerly)
        with context.scope():
            x = tf.fill(dims=(2, 2), value=5.0)
            assert tf.executing_eagerly() == run_eagerly

        assert isinstance(x, tf.Tensor)
        actual_result = context.evaluate(x)
        assert isinstance(actual_result, np.ndarray)

        expected_result = np.array([[5.0, 5.0], [5.0, 5.0]])
        np.testing.assert_equal(actual_result, expected_result)


if __name__ == "__main__":
    unittest.main()
