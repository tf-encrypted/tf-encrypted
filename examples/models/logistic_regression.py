import tensorflow as tf

import tf_encrypted as tfe


def logistic_regression(batch_input_shape, classes, private=True):
    if private:
        model = tfe.keras.Sequential()
        model.add(tfe.keras.layers.Flatten(batch_input_shape=batch_input_shape))
        model.add(tfe.keras.layers.Dense(1, activation=None, lazy_normalization=True))
        return model
    else:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(batch_input_shape=batch_input_shape))
        model.add(tf.keras.layers.Dense(1, activation=None))
        return model
