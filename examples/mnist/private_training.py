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
    EPOCHS = 1

    ITERATIONS = 60000 // BATCH_SIZE

    IMG_ROWS = 28
    IMG_COLS = 28
    FLATTENED_DIM = IMG_ROWS * IMG_COLS

    def __init__(self):
        self.model = tfe.keras.Sequential()
        self.model.add(tfe.keras.layers.Dense(512, batch_input_shape=[self.BATCH_SIZE, self.FLATTENED_DIM]))
        self.model.add(tfe.keras.layers.Activation("relu"))
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

        def flatten(image, label):
            image = tf.reshape(image, shape=[self.FLATTENED_DIM])
            return image, label

        dataset = tf.data.TFRecordDataset([self.local_data_file])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.map(flatten)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)

        iterator = dataset.make_one_shot_iterator()
        x, y = iterator.get_next()
        x = tf.reshape(x, [self.BATCH_SIZE, self.FLATTENED_DIM])
        y = tf.reshape(y, [self.BATCH_SIZE, self.NUM_CLASSES])
        return x, y

    def _train(self, training_data):
        """Build a graph for private model training."""

        x, y = training_data
        self.model.fit(x, y, epochs=self.EPOCHS, steps_per_epoch=self.ITERATIONS)

    def provide_weights(self):
        with tf.name_scope("loading"):
            training_data = self._build_data_pipeline()

        with tf.name_scope("training"):
            self._train(training_data)

        return self.model.weights


class PredictionClient(PrivateModel):
    """
  Contains code meant to be executed by a prediction client.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the data owner
    build_update_step: `Callable`, the function used to construct
                       a local federated learning update.
  """

    BATCH_SIZE = 20

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

        def flatten(image, label):
            image = tf.reshape(image, shape=[self.FLATTENED_DIM])
            return image, label

        dataset = tf.data.TFRecordDataset([self.local_data_file])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.map(flatten)
        dataset = dataset.batch(self.BATCH_SIZE)

        iterator = dataset.make_one_shot_iterator()
        x, y = iterator.get_next()
        x = tf.reshape(x, [self.BATCH_SIZE, self.FLATTENED_DIM])
        y = tf.reshape(y, [self.BATCH_SIZE, self.NUM_CLASSES])
        return x, y

    @tfe.local_computation
    def provide_input(self) -> tf.Tensor:
        """Prepare input data for prediction."""
        with tf.name_scope("loading"):
            prediction_input, expected_result = self._build_data_pipeline().get_next()
            print_op = tf.print("Expect", expected_result, summarize=self.BATCH_SIZE)
            with tf.control_dependencies([print_op]):
                prediction_input = tf.identity(prediction_input)

        with tf.name_scope("pre-processing"):
            prediction_input = tf.reshape(
                prediction_input, shape=(self.BATCH_SIZE, ModelOwner.FLATTENED_DIM)
            )
        return prediction_input

    @tfe.local_computation
    def receive_output(self, logits: tf.Tensor) -> tf.Operation:
        with tf.name_scope("post-processing"):
            prediction = tf.argmax(logits, axis=1)
            op = tf.print("Result", prediction, summarize=self.BATCH_SIZE)
            return op

    def evaluate(self):
        with tf.name_scope("loading"):
            x, y = self._build_data_pipeline()

        with tf.name_scope("evaluate"):
            result = self.model.evaluate(x, y, metrics=["categorical_accuracy"])

        return result


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    sess = tfe.Session(target=session_target)
    KE.set_session(sess)

    training_client = TrainingClient(
        player_name="training-client", local_data_file="./data/train.tfrecord"
    )

    prediction_client = PredictionClient(
        player_name="prediction-client", local_data_file="./data/test.tfrecord"
    )

    # get model parameters as private tensors from training client
    print("Train model")
    params = training_client.provide_weights()

    print("Set trained weights")
    prediction_client.model.set_weights(params)

    print("Evaluate")
    result = prediction_client.evaluate()

    print(result)

