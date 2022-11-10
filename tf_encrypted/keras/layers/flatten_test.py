# pylint: disable=missing-docstring
import unittest

import numpy as np
import pytest

import tf_encrypted as tfe
from tf_encrypted.keras.testing_utils import agreement_test

np.random.seed(42)


@pytest.mark.layers
class TestFlatten(unittest.TestCase):
    def test_flatten_rank_four(self):
        self._core_flatten(batch_input_shape=[4, 5, 2, 2])

    def test_flatten_rank_one(self):
        self._core_flatten(batch_input_shape=[4])

    def test_flatten_channels_first(self):
        self._core_flatten(batch_input_shape=[4, 5, 2, 2], data_format="channels_first")

    def _core_flatten(self, **layer_kwargs):
        batch_input_shape = layer_kwargs["batch_input_shape"]

        agreement_test(
            tfe.keras.layers.Flatten,
            kwargs=layer_kwargs,
            batch_input_shape=batch_input_shape,
            atol=0.1,
        )


if __name__ == "__main__":
    unittest.main()
