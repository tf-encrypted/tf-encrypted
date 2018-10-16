from __future__ import absolute_import
from typing import Optional, Tuple
import abc
import sys
import math

import tensorflow as tf

from .protocol import memoize, Protocol
from ..tensor import AbstractTensor
from ..player import Player
from ..config import get_config


class PlayerState:
    def __init__(self, left, right):
        self.left = left
        self.right = right

class ReplicatedTensor(abc.ABC):
    pass


class ReplicatedPublicTensor(ReplicatedTensor):
    pass


class ReplicatedPrivateTensor(ReplicatedTensor):

    dispatch_id = 'private'

    def __init__(
        self,
        prot: Protocol,
        shares0: Tuple[AbstractTensor, AbstractTensor],
        shares1: Tuple[AbstractTensor, AbstractTensor],
        shares2: Tuple[AbstractTensor, AbstractTensor]
    ) -> None:
        # assert share0.shape.as_list() == share1.shape.as_list() == share2.shape.as_list()

        self.prot = prot
        self.shares0 = shares0
        self.shares1 = shares1
        self.shares2 = shares2

    @property
    def unwrapped(self):
        return (self.shares0, self.shares1, self.shares2)

    # @property
    # def shape(self) -> List[int]:
    #     return self.share0.shape.as_list()


class Replicated(Protocol):

    def __init__(
        self,
        server0: Optional[Player] = None,
        server1: Optional[Player] = None,
        server2: Optional[Player] = None,
    ) -> None:
        self.server0 = server0 or get_config().get_player('server0')
        self.server1 = server1 or get_config().get_player('server1')
        self.server2 = server2 or get_config().get_player('server2')

    def _add_private_private(self, x: ReplicatedPrivateTensor, y: ReplicatedPrivateTensor) -> ReplicatedPrivateTensor:

        with tf.device(self.server0.device_name):
            x1, x2 = x.shares0
            y1, y2 = y.shares0
            z1 = x1 + y1
            z2 = x2 + y2


