from __future__ import absolute_import

import tensorflow as tf

from . import Player

class InputProvider(object):

    def __init__(self, player: Player) -> None:
        self.player = player

    def provide_input(self) -> tf.Tensor:
        raise NotImplementedError()

class OutputReceiver(object):

    def __init__(self, player: Player) -> None:
        self.player = player

    def receive_output(self, tensor: tf.Tensor) -> tf.Operation:
        raise NotImplementedError()