from ..config import RemoteConfig

if __name__ == '__main__':

    import argparse
    import tf_encrypted as tfe

    parser = argparse.ArgumentParser(description="Run a tf-encrypted player")
    parser.add_argument('name', metavar='NAME', type=str, help='name of player as specified in the config file')
    parser.add_argument('--config', metavar='FILE', type=str, help='path to configuration file', default='./config.json')
    args = parser.parse_args()

    config = tfe.config.load(args.config)

    # pylint: disable=E1101
    if isinstance(config, RemoteConfig):
        server = config.server(args.name)
        server.start()
        server.join()
