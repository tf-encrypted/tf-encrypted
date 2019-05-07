"""Implementation of the Player abstraction."""
import tensorflow as tf


class Player:
  """
  An abstraction for players in the game-theoretic threat model of
  a secure computation.

  :param str name: Name of the player
  :param int index: Index of the player (for ordering)
  :param str device_name: Name of device (fully expanded)
  :param str host: IP/domain address of the player's device, defaults to None
  """

  def __init__(self, name, index, device_name, host=None):
    self.name = name
    self.index = index
    self.device_name = device_name
    self.host = host


def player_device(player: Player):
  """
  Retrieves the tf.device associated with a :class:`Player` object.

  :param Player player: The :class:`Player` object.
  """
  return tf.device(player.device_name)
