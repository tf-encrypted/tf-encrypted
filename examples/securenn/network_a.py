# pylint:  disable=redefined-outer-name
"""An example of performing secure inference with MNIST.

Reproduces Network A from SecureNN, Wagh et al.
"""
from __future__ import absolute_import

import sys
from typing import List

import tensorflow as tf
import tensorflow.keras as keras

import tf_encrypted as tfe
from conv_convert import get_data_from_tfrecord

# tfe.set_tfe_events_flag(True)

if len(sys.argv) >= 2:
    # config file was specified
    config_file = sys.argv[1]
    config = tfe.config.load(config_file)
else:
    # default to using local config
    config = tfe.LocalConfig(
        ["server0", "server1", "crypto-producer", "model-trainer", "prediction-client"]
    )
tfe.set_config(config)
players = ["server0", "server1", "crypto-producer"]
prot = tfe.protocol.SecureNN(*tfe.get_config().get_players(players))
tfe.set_protocol(prot)
session_target = sys.argv[2] if len(sys.argv) > 2 else None


class ModelTrainer:
    """Contains code meant to be executed by a model training Player."""

    BATCH_SIZE = 256
    ITERATIONS = 60000 // BATCH_SIZE
    EPOCHS = 3
    LEARNING_RATE = 3e-3
    IN_N = 28 * 28
    HIDDEN_N = 128
    OUT_N = 10

    def cond(
        self, i: int, max_iter: tf.Tensor, nb_epochs: tf.Tensor, avg_loss: tf.Tensor
    ):
        """Check if training termination condition has been met."""
        is_end_epoch = tf.equal(i % max_iter, 0)
        to_continue = tf.cast(i < max_iter * nb_epochs, tf.bool)

        def true_fn() -> tf.Tensor:
            to_continue = tf.print("avg_loss: ", avg_loss)
            return to_continue

        def false_fn() -> tf.Tensor:
            return to_continue

        return tf.cond(is_end_epoch, true_fn, false_fn)

    def build_training_graph(self, training_data) -> List[tf.Tensor]:
        """Build a graph for plaintext model training.

    Returns a list of the trained model's parameters.
    """
        j = self.IN_N
        k = self.HIDDEN_N
        m = self.OUT_N

        # model parameters and initial values
        model = keras.Sequential()
        model.add(keras.layers.Dense(k, input_shape=[j]))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dense(k))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dense(m))

        # optimizer and data pipeline
        optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)

        def loss(model, inputs, targets):
            logits = model(inputs)
            per_element_loss = tf.losses.sparse_softmax_cross_entropy(
                labels=targets, logits=logits
            )
            return tf.reduce_mean(per_element_loss)

        def grad(model, inputs, targets):
            loss_value = loss(model, inputs, targets)
            return loss_value, tf.gradients(loss_value, model.trainable_variables)

        # training loop
        def loop_body(i, max_iter, nb_epochs, avg_loss):
            x, y = training_data.get_next()
            loss, grads = grad(model, x, y)
            update_op = optimizer.apply_gradients(zip(grads, model.trainable_variables))

            is_end_epoch = tf.equal(i % max_iter, 0)

            def true_fn() -> tf.Tensor:
                return loss

            def false_fn() -> tf.Tensor:
                prev_loss = tf.cast(i - 1, tf.float32) * avg_loss
                return (prev_loss + loss) / tf.cast(i, tf.float32)

            with tf.control_dependencies([update_op]):
                terminal_cond = tf.cond(is_end_epoch, true_fn, false_fn)
                return i + 1, max_iter, nb_epochs, terminal_cond

        loop, _, _, _ = tf.while_loop(
            self.cond, loop_body, [0, self.ITERATIONS, self.EPOCHS, 0.0]
        )

        # return model parameters after training
        loop = tf.print("Training complete", loop)

        with tf.control_dependencies([loop]):
            return [tf.identity(x) for x in model.trainable_variables]

    def provide_input(self) -> List[tf.Tensor]:
        with tf.name_scope("loading"):
            training_data = get_data_from_tfrecord(
                "./data/train.tfrecord", self.BATCH_SIZE, flattened=True
            )

        with tf.name_scope("training"):
            parameters = self.build_training_graph(training_data)

        return parameters


class PredictionClient:
    """Contains methods meant to be executed by a prediction client.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the data owner
    build_update_step: `Callable`, the function used to construct
                       a local federated learning update.
  """

    BATCH_SIZE = 20

    def provide_input(self) -> List[tf.Tensor]:
        """Prepare input data for prediction."""
        with tf.name_scope("loading"):
            prediction_input, expected_result = get_data_from_tfrecord(
                "./data/test.tfrecord", self.BATCH_SIZE, flattened=True
            ).get_next()

        with tf.name_scope("pre-processing"):
            prediction_input = tf.reshape(
                prediction_input, shape=(self.BATCH_SIZE, ModelTrainer.IN_N)
            )
            expected_result = tf.reshape(expected_result, shape=(self.BATCH_SIZE,))

        return [prediction_input, expected_result]

    def receive_output(self, likelihoods: tf.Tensor, y_true: tf.Tensor):
        with tf.name_scope("post-processing"):
            prediction = tf.argmax(likelihoods, axis=1)
            eq_values = tf.equal(prediction, tf.cast(y_true, tf.int64))
            acc = tf.reduce_mean(tf.cast(eq_values, tf.float32))
            op = tf.print(
                "Expected:", y_true, "\nActual:", prediction, "\nAccuracy:", acc
            )

            return op


if __name__ == "__main__":
    model_trainer = ModelTrainer()
    prediction_client = PredictionClient()

    # get model parameters as private tensors from model owner
    params = tfe.define_private_input(
        "model-trainer", model_trainer.provide_input
    )  # pylint: disable=E0632

    # we'll use the same parameters for each prediction so we cache them to
    # avoid re-training each time
    cache_updater, params = tfe.cache(params)

    # get prediction input from client
    x, y = tfe.define_private_input(
        "prediction-client", prediction_client.provide_input
    )  # pylint: disable=E0632

    with tfe.protocol.SecureNN():
        batch_size = PredictionClient.BATCH_SIZE
        flat_dim = ModelTrainer.IN_N
        batch_input_shape = [batch_size, flat_dim]
        # compute prediction
        model = tfe.keras.Sequential()
        model.add(
            tfe.keras.layers.Dense(
                ModelTrainer.HIDDEN_N, batch_input_shape=batch_input_shape
            )
        )
        model.add(tfe.keras.layers.Activation("relu"))
        model.add(tfe.keras.layers.Dense(ModelTrainer.HIDDEN_N))
        model.add(tfe.keras.layers.Activation("relu"))
        model.add(tfe.keras.layers.Dense(ModelTrainer.OUT_N))

        logits = model(x)

    # send prediction output back to client
    prediction_op = tfe.define_output(
        "prediction-client", [logits, y], prediction_client.receive_output
    )

    sess = tfe.Session(target=session_target)
    sess.run(tf.global_variables_initializer(), tag="init")

    print("Training")
    sess.run(cache_updater, tag="training")

    print("Set trained weights")
    model.set_weights(params, sess)

    for _ in range(5):
        print("Predicting")
        sess.run(prediction_op, tag="prediction")

    sess.close()
