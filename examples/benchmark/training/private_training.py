# pylint:  disable=redefined-outer-name
"""An example of performing secure training with various model and dataset.
"""
import argparse
import sys

from lr_mnist_dataset import LRMnistDataset  # noqa:F403,F401

import tf_encrypted as tfe
from tf_encrypted.keras.datasets import *  # noqa:F403,F401
from tf_encrypted.player import DataOwner
from tf_encrypted.protocol import ABY3  # noqa:F403,F401
from tf_encrypted.protocol import Pond  # noqa:F403,F401
from tf_encrypted.protocol import SecureNN  # noqa:F403,F401

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
                "training-client",
                "prediction-client",
            ]
        )
        tfe.set_config(config)

    # set tfe protocol
    tfe.set_protocol(globals()[args.protocol](fixedpoint_config=args.precision))

    Dataset = globals()[args.data_name + "Dataset"]
    # set train data owner
    train_dataset = Dataset(batch_size=128)
    training_client = DataOwner(
        config.get_player("training-client"), train_dataset.generator_builder()
    )
    # set test data owner
    test_dataset = Dataset(batch_size=100, train=False)
    prediction_client = DataOwner(
        config.get_player("prediction-client"), test_dataset.generator_builder()
    )

    # set model to be trained
    model = globals()[args.model_name](
        train_dataset.batch_shape, train_dataset.num_classes
    )
    if train_dataset.num_classes > 2:
        loss = tfe.keras.losses.CategoricalCrossentropy(
            from_logits=True, lazy_normalization=True
        )
        metrics = metrics = ["categorical_accuracy"]
    else:
        loss = tfe.keras.losses.BinaryCrossentropy(
            from_logits=True, lazy_normalization=True
        )
        metrics = ["binary_accuracy"]
    # optimizer = tfe.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    # optimizer = tfe.keras.optimizers.AMSgrad(learning_rate=0.001)
    optimizer = tfe.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer, loss)

    print("Train model")
    train_data_iter = training_client.provide_data()
    model.fit(
        x=train_data_iter, epochs=args.epochs, steps_per_epoch=train_dataset.iterations
    )

    print("Set trained weights")
    model_2 = globals()[args.model_name](
        test_dataset.batch_shape, test_dataset.num_classes
    )
    model_2.set_weights(model.weights)

    print("Evaluate")
    test_data_iter = prediction_client.provide_data()
    result = model_2.evaluate(
        x=test_data_iter, metrics=metrics, steps=test_dataset.iterations
    )

    print(result)
