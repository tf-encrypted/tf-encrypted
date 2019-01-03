from __future__ import absolute_import
from typing import Optional, Tuple
import abc
import sys
import math

import tensorflow as tf

from ..protocol import memoize, Protocol
from ...tensor.factory import AbstractTensor
from ...player import Player
from ...config import get_config


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
        prot,
        shares0: Tuple[AbstractTensor, AbstractTensor],
        shares1: Tuple[AbstractTensor, AbstractTensor],
        shares2: Tuple[AbstractTensor, AbstractTensor],
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

    def initializer(self) -> tf.Operation:
        return 


def _private_input(prot, player, input_fn) -> ReplicatedPrivateTensor:



def _add_private_private(prot, x: ReplicatedPrivateTensor, y: ReplicatedPrivateTensor) -> ReplicatedPrivateTensor:

    (x1_0, x2_0), (x0_1, x2_1), (x0_2, x1_2) = x.unwrapped
    (y1_0, y2_0), (y0_1, y2_1), (y0_2, y1_2) = y.unwrapped

    with tf.device(prot.server0.device_name):
        z1_0 = x1_0 + y1_0
        z2_0 = x2_0 + y2_0

    with tf.device(prot.server1.device_name):
        z0_1 = x0_1 + y0_1
        z2_1 = x2_1 + y2_1

    with tf.device(prot.server2.device_name):
        z0_2 = x0_2 + y0_2
        z1_2 = x1_2 + y1_2

    return ReplicatedPrivateTensor(
        prot,
        (z1_0, z2_0),
        (z0_1, z2_1),
        (z0_2, z1_2),
    )
