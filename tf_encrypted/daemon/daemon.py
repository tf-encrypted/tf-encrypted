import json
import logging
import subprocess

import rpyc


server_cmd = "python -m tf_encrypted.player --config {config_file} {player_name}"
search_destroy_cmd = "kill $(ps aux | egrep '[p]ython -m tf_encrypted.player --config' | awk '{print $2}')"


class Daemon(rpyc.Service):
  """
  TF Encrypted daemon for managing player servers remotely.
  """

  def __init__(self):
    self._servers = dict()

  def list_players(self):
    """
    Return list of all players managed by this daemon.
    """
    return list(self._servers.keys())

  def start_player(self, player_name):
    """
    Start a player from the given configuration and player name.

    TODO: provide configuration
    """

    config_file = "/tmp/tfe.config"
    # config.save(config_filename)

    cmd = server_cmd.format(
        config_file=config_file,
        player_name=player_name,
    )
    self._servers[player_name] = subprocess.Popen(cmd.split(' '))

  def stop_player(self, player_name):
    """
    Stop player with given name (if exists).
    """
    if not player_name in self._servers:
      return

    server = self._players[player_name]
    server.kill()
    server.communicate()
    del self._servers[player_name]

  def search_destroy(self):
    """
    Look for anything that looks like a player process and kill it.

    This command is useful for stopping players that have been
    detached from the daemons that launched them.
    """
    subprocess.run(search_destroy_cmd, shell=True, check=True)


def connect_to_daemon(host, port):
  proxy = rpyc.connect(host, port)
  return proxy.root
