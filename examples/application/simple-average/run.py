"""Example of a simple average using TF Encrypted."""
import argparse

import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.protocol import ABY3  # noqa:F403,F401
from tf_encrypted.protocol import Pond  # noqa:F403,F401
from tf_encrypted.protocol import SecureNN  # noqa:F403,F401

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a TF Encrypted model")
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
                "inputter-0",
                "inputter-1",
                "inputter-2",
                "inputter-3",
                "inputter-4",
                "result-receiver",
            ]
        )
        tfe.set_config(config)

    # set tfe protocol
    tfe.set_protocol(globals()[args.protocol]())

    @tfe.local_computation(name_scope="provide_input")
    def provide_input() -> tf.Tensor:
        # pick random tensor to be averaged
        return tf.random.normal(shape=(10,))

    @tfe.local_computation(player_name="result-receiver", name_scope="receive_output")
    def receive_output(average: tf.Tensor):
        # simply print average
        tf.print("Average:", average)

    # get input from inputters as private values
    inputs = [
        provide_input(
            player_name="inputter-0"
        ),  # pylint: disable=unexpected-keyword-arg
        provide_input(
            player_name="inputter-1"
        ),  # pylint: disable=unexpected-keyword-arg
        provide_input(
            player_name="inputter-2"
        ),  # pylint: disable=unexpected-keyword-arg
        provide_input(
            player_name="inputter-3"
        ),  # pylint: disable=unexpected-keyword-arg
        provide_input(
            player_name="inputter-4"
        ),  # pylint: disable=unexpected-keyword-arg
    ]

    # sum all inputs and divide by count
    result = tfe.add_n(inputs) / len(inputs)

    # send result to receiver
    receive_output(result)
