from typing import Optional

import tensorflow as tf


class Player(object):
    def __init__(self, name: str, index: int, device_name: str, host: Optional[str] = None) -> None:
        self.name = name
        self.index = index
        self.device_name = device_name
        self.host = host


def player(player: Player):
    return tf.device(player.device_name)
