from __future__ import absolute_import
import argparse
import logging

import rpyc

from .daemon import Daemon


logger = logging.getLogger('tf_encrypted')


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="Run a TF Encrypted daemon")
  parser.add_argument('--port', '-p',
                      type=int,
                      default=7050,
                      help='TODO')
  args = parser.parse_args()

  logger.info("Launching daemon on port %s", args.port)
  server = rpyc.utils.server.ThreadedServer(
      Daemon,
      port=args.port,
      protocol_config={
          'allow_public_attrs': True,
      },
  )
  server.start()
