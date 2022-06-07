# pylint: disable=missing-docstring
import unittest

import numpy as np
import pytest
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.testing_utils import agreement_test
from tf_encrypted.keras.testing_utils import layer_test

np.random.seed(42)


@pytest.mark.layers
class TestPooling2d(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_maxpooling2d_valid(self):
        self._core_maxpooling2d(strides=2, padding="valid")

    def test_maxpooling2d_same(self):
        self._core_maxpooling2d(strides=2, padding="same")

    def test_maxpooling2d_strides_one(self):
        self._core_maxpooling2d(strides=1, padding="valid")

    def test_avgpooling2d_valid(self):
        self._core_avgpooling2d(strides=2, padding="valid")

    def test_avgpooling2d_same(self):
        self._core_avgpooling2d(strides=2, padding="same")

    def test_avgpooling2d_strides_one(self):
        self._core_avgpooling2d(strides=1, padding="valid")

    def _core_maxpooling2d(self, **layer_kwargs):
        channel_in = 2
        input_shape = [2, 8, 8, channel_in]  # channels last
        pool_size_in = 2

        base_kwargs = {
            "pool_size": pool_size_in,
        }

        kwargs = {**base_kwargs, **layer_kwargs}
        agreement_test(
            tfe.keras.layers.MaxPooling2D, kwargs=kwargs, input_shape=input_shape,
        )
        layer_test(
            tfe.keras.layers.MaxPooling2D, kwargs=kwargs, batch_input_shape=input_shape,
        )

    def _core_avgpooling2d(self, **layer_kwargs):
        channel_in = 2
        input_shape = [2, 8, 8, channel_in]  # channels last
        pool_size_in = 2

        base_kwargs = {
            "pool_size": pool_size_in,
        }

        kwargs = {**base_kwargs, **layer_kwargs}
        agreement_test(
            tfe.keras.layers.AveragePooling2D, kwargs=kwargs, input_shape=input_shape,
        )
        layer_test(
            tfe.keras.layers.AveragePooling2D,
            kwargs=kwargs,
            batch_input_shape=input_shape,
        )


@pytest.mark.layers
class TestGlobalPooling2d(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_global_maxpooling2d(self):
        self._core_global_maxpooling2d()

    def test_global_avgpooling2d(self):
        self._core_global_avgpooling2d()

    def _core_global_maxpooling2d(self, **layer_kwargs):
        channel_in = 2
        input_shape = [2, 4, 4, channel_in]  # channels last

        base_kwargs = {}

        kwargs = {**base_kwargs, **layer_kwargs}
        agreement_test(
            tfe.keras.layers.GlobalMaxPooling2D, kwargs=kwargs, input_shape=input_shape,
        )
        layer_test(
            tfe.keras.layers.GlobalMaxPooling2D,
            kwargs=kwargs,
            batch_input_shape=input_shape,
        )

    def _core_global_avgpooling2d(self, **layer_kwargs):
        channel_in = 2
        input_shape = [2, 4, 4, channel_in]  # channels last

        base_kwargs = {}

        kwargs = {**base_kwargs, **layer_kwargs}
        agreement_test(
            tfe.keras.layers.GlobalAveragePooling2D,
            kwargs=kwargs,
            input_shape=input_shape,
        )
        layer_test(
            tfe.keras.layers.GlobalAveragePooling2D,
            kwargs=kwargs,
            batch_input_shape=input_shape,
        )


if __name__ == "__main__":
    unittest.main()
