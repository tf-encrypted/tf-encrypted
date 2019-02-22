from ..config import RemoteConfig

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Run a TF Encrypted player")
    parser.add_argument('name', metavar='NAME', type=str,
                        help='name of player as specified in the config file')
    parser.add_argument('--config', metavar='FILE', type=str,
                        help='path to configuration file', default='./config.json')
    args = parser.parse_args()

    config = RemoteConfig.load(args.config)
    server = config.server(args.name, start=True)
    server.join()
