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
class TestDense(unittest.TestCase):
    def test_dense_bias(self):
        self._core_dense(use_bias=True)

    def test_dense_nobias(self):
        self._core_dense(use_bias=False)

    def test_dense_relu(self):
        self._core_dense(activation="relu")

    def _core_dense(self, **layer_kwargs):
        batch_input_shape = [4, 5]
        kernel = np.random.normal(size=batch_input_shape[::-1])
        initializer = tf.keras.initializers.Constant(kernel)

        base_kwargs = {
            "units": 4,
            "kernel_initializer": initializer,
        }
        kwargs = {**base_kwargs, **layer_kwargs}
        agreement_test(
            tfe.keras.layers.Dense,
            kwargs=kwargs,
            batch_input_shape=batch_input_shape,
        )
        layer_test(
            tfe.keras.layers.Dense,
            kwargs=kwargs,
            batch_input_shape=batch_input_shape,
        )

    def test_backward(self):
        input_shape = [1, 5]
        input_data = np.random.normal(size=input_shape)
        weights_second_layer = np.random.normal(size=(1, 5))
        kernel = np.random.normal(size=(5, 5))
        initializer = tf.keras.initializers.Constant(kernel)

        with tf.name_scope("TFE"):

            private_input = tfe.define_private_variable(input_data)
            w = tfe.define_private_variable(weights_second_layer)

            tfe_layer = tfe.keras.layers.Dense(
                5,
                input_shape=input_shape[1:],
                kernel_initializer=initializer,
            )

            dense_out_pond = tfe_layer(private_input)

            loss = dense_out_pond * w

            # backward
            grad, d_x = tfe_layer.backward(w)

            tfe_loss = loss.reveal().to_native()
            tfe_d_k = grad[0].reveal().to_native()
            tfe_d_b = grad[1].reveal().to_native()
            tfe_d_x = d_x.reveal().to_native()

        with tf.name_scope("TF"):

            initializer = tf.keras.initializers.Constant(kernel)

            tf_layer = tf.keras.layers.Dense(
                5,
                input_shape=input_shape[1:],
                kernel_initializer=initializer,
            )
            x = tf.Variable(input_data, dtype=tf.float32)
            w = tf.Variable(weights_second_layer, dtype=tf.float32)

            with tf.GradientTape() as tape:
                y = tf_layer(x)
                loss = y * w
                k, b = tf_layer.trainable_weights
                d_x, d_k, d_b = tape.gradient(loss, [x, k, b])

            np.testing.assert_array_almost_equal(tfe_loss, loss, decimal=2)
            np.testing.assert_array_almost_equal(tfe_d_k, d_k, decimal=2)
            np.testing.assert_array_almost_equal(tfe_d_b, d_b, decimal=2)
            np.testing.assert_array_almost_equal(tfe_d_x, d_x, decimal=2)


if __name__ == "__main__":
    unittest.main()
