# pylint: disable=missing-docstring
import unittest

import numpy as np
import pytest
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.testing_utils import agreement_test
from tf_encrypted.keras.testing_utils import layer_test


@pytest.mark.layers
class TestActivation(unittest.TestCase):
    def test_activation_relu(self):
        self._core_activation(activation="relu")

    def _core_activation(self, **layer_kwargs):
        agreement_test(
            tfe.keras.layers.Activation,
            kwargs=layer_kwargs,
            batch_input_shape=[1, 5],
        )
        layer_test(
            tfe.keras.layers.Activation,
            kwargs=layer_kwargs,
            batch_input_shape=[1, 5],
        )

    def test_backward_sigmoid(self) -> None:

        input_shape = [1, 4]
        input_data = np.array([-1, -0.75, 0.75, 1]).reshape(input_shape)
        weights_second_layer = np.ones(shape=[1])

        with tf.name_scope("TFE"):
            private_input = tfe.define_private_variable(input_data)
            w = tfe.define_private_variable(weights_second_layer)

            tfe_layer = tfe.keras.layers.Activation("sigmoid", input_shape=[4])

            dense_out_pond = tfe_layer(private_input)

            loss = dense_out_pond * w

            # backward
            _, d_x = tfe_layer.backward(w)

            tfe_loss = loss.reveal().to_native()
            tfe_d_x = d_x.reveal().to_native()

        with tf.name_scope("TF"):

            tf_layer = tf.keras.layers.Activation("sigmoid", input_shape=[4])

            x = tf.Variable(input_data, dtype=tf.float32)
            w = tf.Variable(weights_second_layer, dtype=tf.float32)

            with tf.GradientTape() as tape:
                y = tf_layer(x)
                loss = y * w

                # backward
                d_x = tape.gradient(loss, x)

            np.testing.assert_array_almost_equal(tfe_loss, loss, decimal=1)
            np.testing.assert_array_almost_equal(tfe_d_x, d_x, decimal=2)


if __name__ == "__main__":
    unittest.main()
