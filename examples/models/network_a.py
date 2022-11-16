import tensorflow as tf

import tf_encrypted as tfe


def network_a(batch_input_shape, classes, private=True):
    if private:
        model = tfe.keras.Sequential()
        model.add(tfe.keras.layers.Flatten(batch_input_shape=batch_input_shape))
        model.add(tfe.keras.layers.Dense(128, activation=None, lazy_normalization=True))
        model.add(tfe.keras.layers.ReLU())
        model.add(tfe.keras.layers.Dense(128, activation=None, lazy_normalization=True))
        model.add(tfe.keras.layers.ReLU())
        model.add(
            tfe.keras.layers.Dense(classes, activation=None, lazy_normalization=True)
        )
        return model
    else:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(batch_input_shape=batch_input_shape))
        model.add(tf.keras.layers.Dense(128, activation=None))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(128, activation=None))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(classes, activation=None))
        return model
