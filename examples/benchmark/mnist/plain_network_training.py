# pylint:  disable=redefined-outer-name
"""An example of performing plain training with MNIST.
"""

import logging
import os
import sys

import tensorflow as tf
from convert import decode


class Model:
    BATCH_SIZE = 128
    NUM_CLASSES = 10
    EPOCHS = 5

    ITERATIONS = 60000 // BATCH_SIZE

    IMG_ROWS = 28
    IMG_COLS = 28
    FLATTENED_DIM = IMG_ROWS * IMG_COLS
    IN_CHANNELS = 1


class NetworkA(Model):
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Flatten(
                batch_input_shape=[
                    self.BATCH_SIZE,
                    self.IMG_ROWS,
                    self.IMG_COLS,
                    self.IN_CHANNELS,
                ]
            )
        )
        self.model.add(
            tf.keras.layers.Dense(128, activation=None)
        )
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(
            tf.keras.layers.Dense(128, activation=None)
        )
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(
            tf.keras.layers.Dense(
                self.NUM_CLASSES, activation=None
            )
        )

        # optimizer and data pipeline
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        # optimizer = tf.keras.optimizers.AMSgrad(learning_rate=0.001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
        )
        self.model.compile(optimizer, loss, metrics=["categorical_accuracy"], )


class NetworkB(Model):
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Conv2D(
                16,
                5,
                1,
                padding="valid",
                activation=None,
                batch_input_shape=[
                    self.BATCH_SIZE,
                    self.IMG_ROWS,
                    self.IMG_COLS,
                    self.IN_CHANNELS,
                ],
            )
        )
        self.model.add(tf.keras.layers.MaxPooling2D(2))
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(
            tf.keras.layers.Conv2D(
                16, 5, 1, padding="valid", activation=None
            )
        )
        self.model.add(tf.keras.layers.MaxPooling2D(2))
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(
            tf.keras.layers.Dense(100, activation=None)
        )
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(
            tf.keras.layers.Dense(
                self.NUM_CLASSES, activation=None
            )
        )

        # optimizer and data pipeline
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        # optimizer = tf.keras.optimizers.AMSgrad(learning_rate=0.001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
        )
        self.model.compile(optimizer, loss, metrics=["categorical_accuracy"], )


class NetworkC(Model):
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Conv2D(
                20,
                5,
                1,
                padding="valid",
                activation=None,
                batch_input_shape=[
                    self.BATCH_SIZE,
                    self.IMG_ROWS,
                    self.IMG_COLS,
                    self.IN_CHANNELS,
                ],
            )
        )
        self.model.add(tf.keras.layers.MaxPooling2D(2))
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(
            tf.keras.layers.Conv2D(
                50, 5, 1, padding="valid", activation=None
            )
        )
        self.model.add(tf.keras.layers.MaxPooling2D(2))
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(
            tf.keras.layers.Dense(500, activation=None)
        )
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(
            tf.keras.layers.Dense(
                self.NUM_CLASSES, activation=None
            )
        )

        # optimizer and data pipeline
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        # optimizer = tf.keras.optimizers.AMSgrad(learning_rate=0.001)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
        )
        self.model.compile(optimizer, loss, metrics=["categorical_accuracy"], )


class NetworkD(Model):
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Conv2D(
                5,
                5,
                2,
                padding="same",
                activation=None,
                batch_input_shape=[
                    self.BATCH_SIZE,
                    self.IMG_ROWS,
                    self.IMG_COLS,
                    self.IN_CHANNELS,
                ],
            )
        )
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(
            tf.keras.layers.Dense(100, activation=None)
        )
        self.model.add(tf.keras.layers.ReLU())
        self.model.add(
            tf.keras.layers.Dense(
                self.NUM_CLASSES, activation=None
            )
        )

        # optimizer and data pipeline
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        # optimizer = tf.keras.optimizers.AMSgrad(learning_rate=0.001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
        )
        self.model.compile(optimizer, loss, metrics=["categorical_accuracy"], )


class TrainingClient(Model):
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

    def _build_data_pipeline(self):
        """Build a reproducible tf.data iterator."""

        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            label = tf.one_hot(label, self.NUM_CLASSES)
            return image, label

        def shaping(image, label):
            image = tf.reshape(
                image,
                shape=[
                    Model.IMG_ROWS,
                    Model.IMG_COLS,
                    Model.IN_CHANNELS,
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
                Model.IMG_ROWS,
                Model.IMG_COLS,
                Model.IN_CHANNELS,
            ],
        )
        y = tf.reshape(y, [self.BATCH_SIZE, self.NUM_CLASSES])
        return x, y

    def train(self, model):
        """Build a graph for model training."""

        with tf.name_scope("loading-data"):
            x, y = self._build_data_pipeline()

        model.fit(x, y, epochs=self.EPOCHS, steps_per_epoch=1)


class PredictionClient(Model):
    """
  Contains code meant to be executed by a prediction client.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the data owner
    build_update_step: `Callable`, the function used to construct
                       a local federated learning update.
  """


    def __init__(self, player_name, local_data_file):
        super().__init__()
        self.player_name = player_name
        self.local_data_file = local_data_file

    def _build_data_pipeline(self):
        """Build a reproducible tf.data iterator."""

        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            label = tf.one_hot(label, self.NUM_CLASSES)
            return image, label

        def shaping(image, label):
            image = tf.reshape(
                image,
                shape=[
                    Model.IMG_ROWS,
                    Model.IMG_COLS,
                    Model.IN_CHANNELS,
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
        return dataset

    def evaluate(self, model):
        with tf.name_scope("loading"):
            dataset = self._build_data_pipeline()

        with tf.name_scope("evaluate"):
            result = model.evaluate(dataset, steps=None)

        return result


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    sess = tf.Session()

    # Network = NetworkA
    # Network = NetworkB
    Network = NetworkC
    # Network = NetworkD

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

    print("Evaluate")
    result = prediction_client.evaluate(model)

    print(result)

