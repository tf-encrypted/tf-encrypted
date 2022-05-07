# pylint:  disable=redefined-outer-name
"""An example of performing secure inference with MNIST.

Also performs plaintext training.
"""

import logging
import sys

import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras import backend as KE
from convert import decode

if len(sys.argv) > 1:
    # config file was specified
    config_file = sys.argv[1]
    config = tfe.RemoteConfig.load(config_file)
    tfe.set_config(config)
else:
    # Always best practice to preset all players to avoid invalid device errors
    config = tfe.LocalConfig(player_names=["server0", "server1", "server2", "training-client", "prediction-client"])
    tfe.set_config(config)

tfe.set_protocol(tfe.protocol.ABY3())

session_target = sys.argv[2] if len(sys.argv) > 2 else None


class PrivateModel:
    BATCH_SIZE = 128
    NUM_CLASSES = 10
    EPOCHS = 5

    ITERATIONS = 60000 // BATCH_SIZE

    IMG_ROWS = 28
    IMG_COLS = 28
    FLATTENED_DIM = IMG_ROWS * IMG_COLS
    IN_CHANNELS = 1

class NetworkA(PrivateModel):
    def __init__(self):
        self.model = tfe.keras.Sequential()
        self.model.add(tfe.keras.layers.Flatten(batch_input_shape=[self.BATCH_SIZE, self.IMG_ROWS, self.IMG_COLS, self.IN_CHANNELS]))
        self.model.add(tfe.keras.layers.Dense(128, activation=None))
        self.model.add(tfe.keras.layers.ReLU())
        self.model.add(tfe.keras.layers.Dense(128, activation=None))
        self.model.add(tfe.keras.layers.ReLU())
        self.model.add(tfe.keras.layers.Dense(self.NUM_CLASSES, activation=None))

        # optimizer and data pipeline
        optimizer = tfe.keras.optimizers.SGD(learning_rate=0.01)
        loss = tfe.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer, loss)

class NetworkB(PrivateModel):
    def __init__(self):
        self.model = tfe.keras.Sequential()
        self.model.add(tfe.keras.layers.Conv2D(16, 5, 1, padding='same', activation=None, batch_input_shape=[self.BATCH_SIZE, self.IMG_ROWS, self.IMG_COLS, self.IN_CHANNELS]))
        self.model.add(tfe.keras.layers.MaxPooling2D(2))
        self.model.add(tfe.keras.layers.ReLU())
        self.model.add(tfe.keras.layers.Conv2D(16, 5, 1, padding='same', activation=None))
        self.model.add(tfe.keras.layers.MaxPooling2D(2))
        self.model.add(tfe.keras.layers.ReLU())
        self.model.add(tfe.keras.layers.Flatten())
        self.model.add(tfe.keras.layers.Dense(100, activation=None))
        self.model.add(tfe.keras.layers.ReLU())
        self.model.add(tfe.keras.layers.Dense(self.NUM_CLASSES, activation=None))

        # optimizer and data pipeline
        optimizer = tfe.keras.optimizers.SGD(learning_rate=0.01)
        loss = tfe.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer, loss)

class NetworkC(PrivateModel):
    def __init__(self):
        self.model = tfe.keras.Sequential()
        self.model.add(tfe.keras.layers.Conv2D(20, 5, 1, padding='valid', activation=None, batch_input_shape=[self.BATCH_SIZE, self.IMG_ROWS, self.IMG_COLS, self.IN_CHANNELS], lazy_normalization=True))
        self.model.add(tfe.keras.layers.MaxPooling2D(2))
        self.model.add(tfe.keras.layers.ReLU())
        self.model.add(tfe.keras.layers.Conv2D(50, 5, 1, padding='valid', activation=None, lazy_normalization=True))
        self.model.add(tfe.keras.layers.MaxPooling2D(2))
        self.model.add(tfe.keras.layers.ReLU())
        self.model.add(tfe.keras.layers.Flatten())
        self.model.add(tfe.keras.layers.Dense(500, activation=None, lazy_normalization=True))
        self.model.add(tfe.keras.layers.ReLU())
        self.model.add(tfe.keras.layers.Dense(self.NUM_CLASSES, activation=None, lazy_normalization=True))

        # optimizer and data pipeline
        optimizer = tfe.keras.optimizers.SGDWithMomentum(learning_rate=0.01, momentum=0.9)
        loss = tfe.keras.losses.CategoricalCrossentropy(from_logits=True, lazy_normalization=True)
        self.model.compile(optimizer, loss)


class NetworkD(PrivateModel):
    def __init__(self):
        self.model = tfe.keras.Sequential()
        self.model.add(tfe.keras.layers.Conv2D(5, 5, 2, padding='same', activation=None, batch_input_shape=[self.BATCH_SIZE, self.IMG_ROWS, self.IMG_COLS, self.IN_CHANNELS]))
        self.model.add(tfe.keras.layers.ReLU())
        self.model.add(tfe.keras.layers.Flatten())
        self.model.add(tfe.keras.layers.Dense(100, activation=None))
        self.model.add(tfe.keras.layers.ReLU())
        self.model.add(tfe.keras.layers.Dense(self.NUM_CLASSES, activation=None))

        # optimizer and data pipeline
        optimizer = tfe.keras.optimizers.SGD(learning_rate=0.01)
        loss = tfe.keras.losses.CategoricalCrossentropy(from_logits=True)
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
            label = tf.one_hot(label, self.NUM_CLASSES)
            return image, label

        def shaping(image, label):
            image = tf.reshape(image, shape=[PrivateModel.IMG_ROWS, PrivateModel.IMG_COLS, PrivateModel.IN_CHANNELS])
            return image, label

        dataset = tf.data.TFRecordDataset([self.local_data_file])
        dataset = dataset \
            .map(decode) \
            .map(normalize) \
            .map(shaping) \
            .cache() \
            .shuffle(60000, reshuffle_each_iteration=True) \
            .repeat() \
            .batch(self.BATCH_SIZE, drop_remainder=True) # drop remainder because we need to fix batch size in private model
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        iterator = dataset.make_one_shot_iterator()
        x, y = iterator.get_next()
        x = tf.reshape(x, [self.BATCH_SIZE, PrivateModel.IMG_ROWS, PrivateModel.IMG_COLS, PrivateModel.IN_CHANNELS])
        y = tf.reshape(y, [self.BATCH_SIZE, self.NUM_CLASSES])
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
            label = tf.one_hot(label, self.NUM_CLASSES)
            return image, label

        def shaping(image, label):
            image = tf.reshape(image, shape=[PrivateModel.IMG_ROWS, PrivateModel.IMG_COLS, PrivateModel.IN_CHANNELS])
            return image, label

        dataset = tf.data.TFRecordDataset([self.local_data_file])
        dataset = dataset \
            .map(decode) \
            .map(normalize) \
            .map(shaping) \
            .cache() \
            .batch(self.BATCH_SIZE, drop_remainder=True) # drop remainder because we need to fix batch size in private model
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        iterator = dataset.make_one_shot_iterator()
        x, y = iterator.get_next()
        x = tf.reshape(x, [self.BATCH_SIZE, PrivateModel.IMG_ROWS, PrivateModel.IMG_COLS, PrivateModel.IN_CHANNELS])
        y = tf.reshape(y, [self.BATCH_SIZE, self.NUM_CLASSES])
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

    # Network = NetworkA
    Network = NetworkC

    model = Network().model

    training_client = TrainingClient(
        player_name="training-client", local_data_file="./data/train.tfrecord"
    )

    prediction_client = PredictionClient(
        player_name="prediction-client", local_data_file="./data/test.tfrecord"
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


