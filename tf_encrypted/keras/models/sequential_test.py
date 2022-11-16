# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras import Sequential
from tf_encrypted.keras.layers import Dense

tf.keras.utils.set_random_seed(42)


class TestSequential(unittest.TestCase):
    def test_two_layers(self):
        shape = (1, 3)
        input_data = np.random.normal(size=shape)
        with tf.name_scope("TFE"):
            model = Sequential()
            model.add(Dense(2, batch_input_shape=shape))
            model.add(Dense(3))

            x = tfe.define_private_variable(input_data)
            model(x)

    def test_model_from_config(self):
        input_shape = (1, 3)
        input_data = np.random.normal(size=input_shape)
        expected, k_weights, k_config = _model_predict_keras(input_data, input_shape)

        with tf.name_scope("TFE"):
            x = tfe.define_private_input(
                "inputter", lambda: tf.convert_to_tensor(input_data)
            )

            tfe_model = tfe.keras.models.model_from_config(k_config)
            tfe_model.set_weights(k_weights)
            y = tfe_model(x)
            actual = y.reveal().to_native()

            np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)

    def test_from_config(self):
        input_shape = (1, 3)
        input_data = np.random.normal(size=input_shape)
        expected, k_weights, k_config = _model_predict_keras(input_data, input_shape)

        with tf.name_scope("TFE"):
            x = tfe.define_private_input(
                "inputter", lambda: tf.convert_to_tensor(input_data)
            )

            tfe_model = Sequential.from_config(k_config)
            tfe_model.set_weights(k_weights)
            y = tfe_model(x)
            actual = y.reveal().to_native()

            np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)

    def test_clone_model(self):
        input_shape = (1, 3)
        input_data = np.random.normal(size=input_shape)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(2, batch_input_shape=input_shape))
        model.add(tf.keras.layers.Dense(3))
        expected = model(input_data)

        with tf.name_scope("TFE"):
            x = tfe.define_private_input(
                "inputter", lambda: tf.convert_to_tensor(input_data)
            )

            tfe_model = tfe.keras.models.clone_model(model)
            y = tfe_model(x)
            actual = y.reveal().to_native()

            np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)

    def test_weights_as_private_var(self):
        input_shape = (1, 3)
        input_data = np.random.normal(size=input_shape)
        expected, k_weights, k_config = _model_predict_keras(input_data, input_shape)

        with tf.name_scope("TFE"):
            x = tfe.define_private_input(
                "inputter", lambda: tf.convert_to_tensor(input_data)
            )

            tfe_model = tfe.keras.models.model_from_config(k_config)
            weights_private_var = [tfe.define_private_variable(w) for w in k_weights]

            tfe_model.set_weights(weights_private_var)
            y = tfe_model(x)
            actual = y.reveal().to_native()

            np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)

    def test_conv_model(self):
        num_classes = 10
        input_shape = (1, 28, 28, 1)
        input_data = np.random.normal(size=input_shape)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(2, (3, 3), batch_input_shape=input_shape))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.AveragePooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(2, (3, 3)))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.AveragePooling2D((2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(num_classes, name="logit"))

        expected = model.predict(input_data)
        k_weights = model.get_weights()
        k_config = model.get_config()

        with tf.name_scope("TFE"):
            x = tfe.define_private_input(
                "inputter", lambda: tf.convert_to_tensor(input_data)
            )

            tfe_model = tfe.keras.models.model_from_config(k_config)

            tfe_model.set_weights(k_weights)
            y = tfe_model(x)
            actual = y.reveal().to_native()

            np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)


class TestSequentialABY3(unittest.TestCase):
    def test_two_layer_training(self):
        shape = (128, 3)
        with tfe.protocol.ABY3():
            tfe_model = Sequential()
            tfe_model.add(Dense(2, batch_input_shape=shape, activation="sigmoid"))
            tfe_model.add(Dense(1))

            x = tfe.define_private_variable(np.random.normal(size=shape))
            y = tfe.define_private_variable(np.random.normal(size=(shape[0], 1)))

            loss = tfe.keras.losses.BinaryCrossentropy(from_logits=True)
            optimizer = tfe.keras.optimizers.SGD(learning_rate=0.01)
            tfe_model.compile(optimizer, loss)
            tfe_model(x)

            weights0_init = [
                w.read_value().reveal().to_native() for w in tfe_model.layers[0].weights
            ]
            weights1_init = [
                w.read_value().reveal().to_native() for w in tfe_model.layers[1].weights
            ]

            loss = tfe_model.fit_batch(x, y)

            weights0_updated = [
                w.read_value().reveal().to_native() for w in tfe_model.layers[0].weights
            ]
            weights1_updated = [
                w.read_value().reveal().to_native() for w in tfe_model.layers[1].weights
            ]

            x_value = x.reveal().to_native()
            y_value = y.reveal().to_native()

        plain_model = tf.keras.Sequential()
        plain_model.add(
            tf.keras.layers.Dense(2, batch_input_shape=shape, activation="sigmoid")
        )
        plain_model.add(tf.keras.layers.Dense(1))

        plain_model.layers[0].set_weights(weights0_init)
        plain_model.layers[1].set_weights(weights1_init)

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        plain_model.compile(optimizer=optimizer, loss=loss)

        plain_model.fit(x_value, y_value, batch_size=128)

        expected_weights0_updated = plain_model.layers[0].weights
        expected_weights1_updated = plain_model.layers[1].weights

        for i in range(len(weights0_updated)):
            np.testing.assert_allclose(
                weights0_updated[i], expected_weights0_updated[i], rtol=1e-3, atol=1e-3
            )
        for i in range(len(weights1_updated)):
            np.testing.assert_allclose(
                weights1_updated[i], expected_weights1_updated[i], rtol=1e-3, atol=1e-3
            )


def _model_predict_keras(input_data, input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(2, batch_input_shape=input_shape))
    model.add(tf.keras.layers.Dense(3))

    weights = model.get_weights()
    config = model.get_config()
    out = model(input_data)

    return out, weights, config


if __name__ == "__main__":
    unittest.main()
