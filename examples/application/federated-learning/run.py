"""An example of the secure aggregation protocol for federated learning."""
import argparse
import sys

import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.datasets import *  # noqa:F403,F401
from tf_encrypted.protocol import ABY3  # noqa:F403,F401
from tf_encrypted.protocol import Pond  # noqa:F403,F401
from tf_encrypted.protocol import SecureNN  # noqa:F403,F401


class ModelOwner:
    """Contains code meant to be executed by some `ModelOwner` Player.

    Args:
      player_name: `str`, name of the `tfe.player.Player`, representing the model owner.
      model_name: `str`, name of the model to be trained
      data_name: `str`, name of dataset that model trained on
    """

    LEARNING_RATE = 0.1
    BATCH_SIZE = 100

    def __init__(self, player_name, model_name, data_name):
        self.player = tfe.get_config().get_player(player_name)
        self.model_name = model_name
        self.data_name = data_name
        self.test_dataset = globals()[self.data_name + "Dataset"](
            batch_size=self.BATCH_SIZE, train=False
        )
        self._build_model()
        self._build_update_func()

    def _build_model(self):
        with tf.device(self.player.device_name):
            self.model = globals()[self.model_name](
                self.test_dataset.batch_shape,
                self.test_dataset.num_classes,
                private=False,
            )
            if self.test_dataset.num_classes > 1:
                self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                self.metrics = lambda y_true, y_pred: tf.reduce_mean(
                    tf.keras.metrics.categorical_accuracy(y_true, y_pred)
                )
            else:
                self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                self.metrics = lambda y_true, y_pred: tf.reduce_mean(
                    tf.keras.metrics.binary_accuracy(y_true, y_pred)
                )
            self.model.compile(loss=self.loss)

    def _build_update_func(self):
        @tfe.local_computation(player_name=self.player.name)
        def update_func(*grads):
            weights = self.model.weights
            grads = [tf.cast(grad, tf.float32) for grad in grads]
            with tf.name_scope("update_model"):
                for weight, grad in zip(weights, grads):
                    weight.assign(weight.read_value() - grad * self.LEARNING_RATE)

        self.update_func = update_func

    def share_weights(self):
        return self.model.weights

    def update_model(self, *grads):
        """Perform a single update step.

        This will be performed on the ModelOwner device
        after securely aggregating gradients.

        Args:
          *grads: `tfe.Tensor` representing the federally computed gradients.
        """
        self.update_func(*grads)

    def validate_model(self):
        with tf.device(self.player.device_name):
            with tf.name_scope("data_loading"):
                data_generator = self.test_dataset.generator_builder()
                data_iter = data_generator()

            # this maybe a bug in tensorflow
            # when using `model.fit` or `model.evalute` within tf.device context
            # program will stop
            with tf.name_scope("validate_model"):
                y_preds = []
                y_trues = []
                for index, (input_x, input_y) in enumerate(data_iter):
                    y_pred = self.model(input_x)
                    y_preds.append(y_pred)
                    y_trues.append(input_y)
                    if index + 1 >= self.test_dataset.iterations:
                        break

                y_pred = tf.concat(y_preds, axis=0)
                y_true = tf.concat(y_trues, axis=0)
                tf.print(self.metrics(y_true, y_pred))


class DataOwner:
    """Contains methods meant to be executed by a data owner.

    Args:
      player_name: `str`, name of the `tfe.player.Player`
                   representing the data owner
      model_name: `str`, name of the model to be trained
      data_name: `str`, name of dataset that model trained on
      data_slice: specify which part of this dataset to use
    """

    BATCH_SIZE = 128

    def __init__(self, player_name, model_name, data_name, data_slice):
        self.player = tfe.get_config().get_player(player_name)
        self.model_name = model_name
        self.data_name = data_name
        self._build_data(data_slice)
        self._build_model()
        self._build_gradient_func()
        self.iterations = self.train_dataset.iterations

    def _build_data(self, data_slice):
        with tf.device(self.player.device_name):
            self.train_dataset = globals()[self.data_name + "Dataset"](
                batch_size=self.BATCH_SIZE
            )[data_slice]
            data_generator = self.train_dataset.generator_builder()
            self.data_iter = data_generator()

    def _build_model(self):
        with tf.device(self.player.device_name):
            self.model = globals()[self.model_name](
                self.train_dataset.batch_shape,
                self.train_dataset.num_classes,
                private=False,
            )
            if self.train_dataset.num_classes > 1:
                self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                self.metrics = ["categorical_accuracy"]
            else:
                self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                self.metrics = ["binary_accuracy"]
            self.model.compile(loss=self.loss, metrics=self.metrics)

    def _build_gradient_func(self):
        @tfe.local_computation(player_name=self.player.name)
        def compute_gradient():
            with tf.name_scope("local_training"):
                with tf.GradientTape() as tape:
                    x, y = next(self.data_iter)
                    y_pred = self.model.call(x)
                    loss = self.loss(y, y_pred)
                gradients = tape.gradient(loss, self.model.trainable_variables)

            return gradients

        self.gradient_func = compute_gradient

    def compute_gradient(self):
        """Compute gradient given current model parameters and local data."""
        return self.gradient_func()

    def update_model(self, *update_weights):
        """update model with new weights from model owner"""
        with tf.device(self.player.device_name):
            weights = self.model.weights
            with tf.name_scope("update"):
                for weight, update_weight in zip(weights, update_weights):
                    weight.assign(update_weight)


@tf.function
def federated_training(model_owner, data_owners):
    # share model owner's model weights to data owners
    update_weights = model_owner.share_weights()
    for data_owner in data_owners:
        data_owner.update_model(*update_weights)
    # collect encrypted gradients from data owners
    model_grads = zip(*(data_owner.compute_gradient() for data_owner in data_owners))
    # compute mean of gradients (without decrypting)
    with tf.name_scope("secure_aggregation"):
        aggregated_model_grads = [
            tfe.add_n(grads) / len(grads) for grads in model_grads
        ]
    # send the encrypted aggregated gradients
    # to the model owner for it to decrypt and update
    model_owner.update_model(*aggregated_model_grads)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a TF Encrypted model")
    parser.add_argument(
        "model_name",
        metavar="MODEL NAME",
        type=str,
        help="name of model to be trained",
    )
    parser.add_argument(
        "data_name",
        metavar="DATASET NAME",
        type=str,
        help="name of dataset which model trained on",
    )
    parser.add_argument(
        "--epochs",
        metavar="EPOCHS",
        type=int,
        default=5,
        help="how many epochs to run",
    )
    parser.add_argument(
        "--protocol",
        metavar="PROTOCOL",
        type=str,
        default="ABY3",
        help="MPC protocol TF Encrypted used",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=str,
        default="./config.json",
        help="path to configuration file",
    )
    args = parser.parse_args()

    # import all models
    sys.path.append("../../")
    from models import *  # noqa:F403,F401

    # set tfe config
    if args.config != "local":
        # config file was specified
        config_file = args.config
        config = tfe.RemoteConfig.load(config_file)
        config.connect_servers()
        tfe.set_config(config)
    else:
        # Always best practice to preset all players to avoid invalid device errors
        config = tfe.LocalConfig(
            player_names=[
                "server0",
                "server1",
                "server2",
                "model-owner",
                "train-data-owner-0",
                "train-data-owner-1",
                "train-data-owner-2",
            ]
        )
        tfe.set_config(config)

    # set tfe protocol
    tfe.set_protocol(globals()[args.protocol]())

    # set model owner and data owners
    model_owner = ModelOwner("model-owner", args.model_name, args.data_name)
    data_owners = [
        DataOwner(
            "train-data-owner-0", args.model_name, args.data_name, slice(0, 20000)
        ),
        DataOwner(
            "train-data-owner-1", args.model_name, args.data_name, slice(20000, 40000)
        ),
        DataOwner(
            "train-data-owner-2", args.model_name, args.data_name, slice(40000, 60000)
        ),
    ]

    # run federated learning
    for e in range(args.epochs):
        for i in range(data_owners[0].iterations):
            federated_training(model_owner, data_owners)
        model_owner.validate_model()
