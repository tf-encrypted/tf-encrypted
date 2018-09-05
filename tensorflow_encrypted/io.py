from __future__ import absolute_import
from typing import Optional, Callable

import tensorflow as tf

from .player import Player


class InputProvider(object):

    def __init__(self, player: Player, input_fn: Optional[Callable[[], tf.Tensor]] = None) -> None:
        self.player = player
        self.input_fn = input_fn

    def provide_input(self) -> tf.Tensor:
        if self.input_fn is None:
            raise NotImplementedError()

        return self.input_fn()


class OutputReceiver(object):

    def __init__(self, player: Player, output_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None) -> None:
        self.player = player
        self.output_fn = output_fn

    def receive_output(self, tensor: tf.Tensor) -> tf.Tensor:
        if self.output_fn is None:
            raise NotImplementedError()

        return self.output_fn(tensor)
