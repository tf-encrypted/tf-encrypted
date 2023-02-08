# pylint:  disable=redefined-outer-name
"""An example of performing secure joint inference
   with model owner and prediction client."""
import argparse
import sys

import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.datasets import *  # noqa:F403,F401
from tf_encrypted.player import DataOwner
from tf_encrypted.protocol import ABY3  # noqa:F403,F401
from tf_encrypted.protocol import Pond  # noqa:F403,F401
from tf_encrypted.protocol import SecureNN  # noqa:F403,F401

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Perform a secure joint inference\
        between model owner and prediction client"
    )
    parser.add_argument(
        "model_name",
        metavar="MODEL NAME",
        type=str,
        help="name of model to be inferenced",
    )
    parser.add_argument(
        "data_name",
        metavar="DATASET NAME",
        type=str,
        help="name of dataset which model trained on",
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
    parser.add_argument(
        "--precision",
        choices=["l", "h", "low", "high"],
        type=str,
        default="l",
        help="use 64 or 128 bits for computation",
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
                "prediction-client",
            ]
        )
        tfe.set_config(config)

    # set tfe protocol
    tfe.set_protocol(globals()[args.protocol](fixedpoint_config=args.precision))

    @tfe.local_computation(player_name="model-owner", name_scope="share_model_weights")
    def share_model_weights(model_name, data_name):
        # model owner train a model and share its weights
        Dataset = globals()[data_name + "Dataset"]
        train_dataset = Dataset(batch_size=128)
        data_iter = train_dataset.generator_builder()()
        model = globals()[model_name](
            train_dataset.batch_shape, train_dataset.num_classes, private=False
        )
        if train_dataset.num_classes > 1:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics = ["categorical_accuracy"]
        else:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            metrics = ["binary_accuracy"]
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        # optimizer = tf.keras.optimizers.AMSgrad(learning_rate=0.001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer, loss, metrics=metrics)
        for i in range(train_dataset.iterations):
            data = next(data_iter)
            model.train_step(data)
        return model.weights

    @tfe.local_computation(
        player_name="prediction-client", name_scope="receive_prediction"
    )
    def receive_prediction(prediction):
        # simply print prediction result
        prediction = tf.nn.softmax(prediction)
        tf.print("Prediction result:", prediction)

    Dataset = globals()[args.data_name + "Dataset"]
    # set prediction client
    test_dataset = Dataset(batch_size=100, train=False)
    prediction_client = DataOwner(
        config.get_player("prediction-client"),
        test_dataset.generator_builder(label=False),
    )

    # share model weihgts
    model_weights = share_model_weights(args.model_name, args.data_name)

    print("Set model weights")
    model = globals()[args.model_name](
        test_dataset.batch_shape, test_dataset.num_classes
    )
    model.set_weights(model_weights)

    print("perform predict")
    result = model.predict(x=prediction_client.provide_data(), reveal=False)
    receive_prediction(result)
