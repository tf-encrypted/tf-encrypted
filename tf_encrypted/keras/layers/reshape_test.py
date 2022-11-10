# pylint: disable=missing-docstring
import unittest

import numpy as np
import pytest

import tf_encrypted as tfe
from tf_encrypted.keras.testing_utils import agreement_test
from tf_encrypted.keras.testing_utils import layer_test

np.random.seed(42)


@pytest.mark.layers
class TestReshape(unittest.TestCase):
    def test_reshape(self):
        self._core_reshape((2, 2, 2, 2), target_shape=[8])

    def test_reshape_unknown_dim(self):
        self._core_reshape((2, 2, 2, 2), target_shape=[-1, 4])

    def _core_reshape(self, batch_input_shape, **layer_kwargs):

        kwargs = {**layer_kwargs}
        agreement_test(
            tfe.keras.layers.Reshape,
            kwargs=kwargs,
            batch_input_shape=batch_input_shape,
        )
        layer_test(
            tfe.keras.layers.Reshape,
            kwargs=kwargs,
            batch_input_shape=batch_input_shape,
        )


if __name__ == "__main__":
    unittest.main()
