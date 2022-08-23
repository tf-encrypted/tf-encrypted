# pylint: disable=missing-docstring
import unittest

import pytest
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.testing_utils import agreement_test


@pytest.mark.layers
class TestActivation(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_activation_relu(self):
        self._core_relu()

    def _core_relu(self, **layer_kwargs):
        agreement_test(
            tfe.keras.layers.ReLU, kwargs=layer_kwargs, input_shape=[1, 5], rtol=0.1,
        )


if __name__ == "__main__":
    unittest.main()
