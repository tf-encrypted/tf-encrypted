# pylint:  disable=redefined-outer-name
"""An example of performing secure inference with MNIST.

Also performs plaintext training.
"""

import logging
import os
import sys

import tensorflow as tf

import tf_encrypted as tfe
from convert import decode
from tf_encrypted.keras import backend as KE

if len(sys.argv) > 1:
    # config file was specified
    config_file = sys.argv[1]
    config = tfe.RemoteConfig.load(config_file)
    tfe.set_config(config)
else:
    # Always best practice to preset all players to avoid invalid device errors
    config = tfe.LocalConfig(
        player_names=[
            "server0",
            "server1",
            "server2",
            "training-client",
            "prediction-client",
        ]
    )
    tfe.set_config(config)

tfe.set_protocol(tfe.protocol.ABY3())

session_target = sys.argv[2] if len(sys.argv) > 2 else None


class PrivateModel:
    BATCH_SIZE = 128
    EPOCHS = 5

    ITERATIONS = 60000 // BATCH_SIZE

    IMG_ROWS = 28
    IMG_COLS = 28
    FLATTENED_DIM = IMG_ROWS * IMG_COLS
    IN_CHANNELS = 1


class LogisticRegression(PrivateModel):
    def __init__(self):
        self.model = tfe.keras.Sequential()
        self.model.add(
            tfe.keras.layers.Flatten(
                batch_input_shape=[
                    self.BATCH_SIZE,
                    self.IMG_ROWS,
                    self.IMG_COLS,
                    self.IN_CHANNELS,
                ]
            )
        )
        self.model.add(
            tfe.keras.layers.Dense(1, activation=None, lazy_normalization=True)
        )

        # optimizer and data pipeline
        # optimizer = tfe.keras.optimizers.SGDWithMomentum(learning_rate=0.01)
        # optimizer = tfe.keras.optimizers.AMSgrad(learning_rate=0.001)
        optimizer = tfe.keras.optimizers.Adam(learning_rate=0.001)
        loss = tfe.keras.losses.BinaryCrossentropy(
            from_logits=True, lazy_normalization=True
        )
        self.model.compile(optimizer, loss)


class TrainingClient(PrivateModel):
    """Contains code meant to be executed by a training client.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the model owner.
    local_data_file: filepath to MNIST data.
  """

    def __init__(self, player_name, local_data_file):
        super().__init__()
        self.player_name = player_name
        self.local_data_file = local_data_file

    @tfe.local_computation
    def _build_data_pipeline(self):
        """Build a reproducible tf.data iterator."""

        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            label = tf.cast(tf.math.greater(label, 4), dtype=tf.float32)
            return image, label

        def shaping(image, label):
            image = tf.reshape(
                image,
                shape=[
                    PrivateModel.IMG_ROWS,
                    PrivateModel.IMG_COLS,
                    PrivateModel.IN_CHANNELS,
                ],
            )
            return image, label

        dataset = tf.data.TFRecordDataset([self.local_data_file])
        dataset = (
            dataset.map(decode)
            .map(normalize)
            .map(shaping)
            .cache()
            .shuffle(60000, reshuffle_each_iteration=True)
            .repeat()
            .batch(self.BATCH_SIZE, drop_remainder=True)
        )  # drop remainder because we need to fix batch size in private model
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        iterator = dataset.make_one_shot_iterator()
        x, y = iterator.get_next()
        x = tf.reshape(
            x,
            [
                self.BATCH_SIZE,
                PrivateModel.IMG_ROWS,
                PrivateModel.IMG_COLS,
                PrivateModel.IN_CHANNELS,
            ],
        )
        y = tf.reshape(y, [self.BATCH_SIZE, 1])
        return x, y

    def train(self, model):
        """Build a graph for private model training."""

        with tf.name_scope("loading-data"):
            x, y = self._build_data_pipeline()

        model.fit(x, y, epochs=self.EPOCHS, steps_per_epoch=self.ITERATIONS)


class PredictionClient(PrivateModel):
    """
  Contains code meant to be executed by a prediction client.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the data owner
    build_update_step: `Callable`, the function used to construct
                       a local federated learning update.
  """

    BATCH_SIZE = 100

    def __init__(self, player_name, local_data_file):
        super().__init__()
        self.player_name = player_name
        self.local_data_file = local_data_file

    def _build_data_pipeline(self):
        """Build a reproducible tf.data iterator."""

        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            label = tf.cast(tf.math.greater(label, 4), dtype=tf.float32)
            return image, label

        def shaping(image, label):
            image = tf.reshape(
                image,
                shape=[
                    PrivateModel.IMG_ROWS,
                    PrivateModel.IMG_COLS,
                    PrivateModel.IN_CHANNELS,
                ],
            )
            return image, label

        dataset = tf.data.TFRecordDataset([self.local_data_file])
        dataset = (
            dataset.map(decode)
            .map(normalize)
            .map(shaping)
            .cache()
            .batch(self.BATCH_SIZE, drop_remainder=True)
        )  # drop remainder because we need to fix batch size in private model
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        iterator = dataset.make_one_shot_iterator()
        x, y = iterator.get_next()
        x = tf.reshape(
            x,
            [
                self.BATCH_SIZE,
                PrivateModel.IMG_ROWS,
                PrivateModel.IMG_COLS,
                PrivateModel.IN_CHANNELS,
            ],
        )
        y = tf.reshape(y, [self.BATCH_SIZE, 1])
        return x, y

    def evaluate(self, model):
        with tf.name_scope("loading"):
            x, y = self._build_data_pipeline()

        with tf.name_scope("evaluate"):
            result = model.evaluate(x, y, metrics=["categorical_accuracy"], steps=None)

        return result


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    sess = tfe.Session(target=session_target)
    KE.set_session(sess)

    Network = LogisticRegression

    model = Network().model

    directory = os.path.dirname(os.path.abspath(__file__))
    train_data_file = os.path.join(directory, "data", "train.tfrecord")
    training_client = TrainingClient(
        player_name="training-client", local_data_file=train_data_file
    )

    test_data_file = os.path.join(directory, "data", "test.tfrecord")
    prediction_client = PredictionClient(
        player_name="prediction-client", local_data_file=test_data_file
    )

    print("Train model")
    training_client.train(model)
    weights = model.weights

    print("Set trained weights")
    model_2 = Network().model
    model_2.set_weights(weights)

    print("Evaluate")
    result = prediction_client.evaluate(model_2)

    print(result)
