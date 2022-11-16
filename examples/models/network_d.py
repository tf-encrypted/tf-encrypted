import tensorflow as tf

import tf_encrypted as tfe


def network_d(batch_input_shape, classes, private=True):
    if private:
        model = tfe.keras.Sequential()
        model.add(
            tfe.keras.layers.Conv2D(
                5,
                5,
                2,
                padding="same",
                activation=None,
                batch_input_shape=batch_input_shape,
                lazy_normalization=True,
            )
        )
        model.add(tfe.keras.layers.ReLU())
        model.add(tfe.keras.layers.Flatten())
        model.add(tfe.keras.layers.Dense(100, activation=None, lazy_normalization=True))
        model.add(tfe.keras.layers.ReLU())
        model.add(
            tfe.keras.layers.Dense(classes, activation=None, lazy_normalization=True)
        )
        return model
    else:
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                5,
                5,
                2,
                padding="same",
                activation=None,
                batch_input_shape=batch_input_shape,
            )
        )
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(100, activation=None))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(classes, activation=None))
        return model
