"""Executable for hosting a Player"""
import logging

from tf_encrypted.config import RemoteConfig

if __name__ == "__main__":

    logging.basicConfig()
    logger = logging.getLogger("tf_encrypted")
    logger.setLevel(logging.DEBUG)

    import argparse

    parser = argparse.ArgumentParser(description="Run a TF Encrypted player")
    parser.add_argument(
        "name",
        metavar="NAME",
        type=str,
        help="name of player as specified in the config file",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=str,
        default="./config.json",
        help="path to configuration file",
    )
    args = parser.parse_args()

    config = RemoteConfig.load(args.config)
    server = config.server(args.name, start=True)
    server.join()
