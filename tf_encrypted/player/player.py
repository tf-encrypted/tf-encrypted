from typing import Optional

import tensorflow as tf


class Player(object):
    """
    An abstraction for players in the game-theoretic model of a secure computation.

    :param str name: Name of the player
    :param int index: Index of the player (for ordering)
    :param str device_name: Name of device (fully expanded)
    :param str host: IP/domain address of the player's device, defaults to None
    """
    def __init__(self, name: str, index: int, device_name: str, host: Optional[str] = None) -> None:
        self.name = name
        self.index = index
        self.device_name = device_name
        self.host = host


def player(player: Player):
    """
    Retrieves the tf.device associated with a :class:`Player` object.

    :param Player player: The :class:`Player` object.
    """
    return tf.device(player.device_name)
