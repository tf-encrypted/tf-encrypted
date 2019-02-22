from __future__ import absolute_import
from typing import Optional, Tuple, List
import abc
import sys
import math

import tensorflow as tf

from ..protocol import memoize, Protocol
from ...tensor.factory import AbstractTensor
from ...player import Player
from ...config import get_config


class ReplicatedTensor(abc.ABC):
    pass

class ReplicatedPublicTensor(ReplicatedTensor):
    pass

class ReplicatedPrivateTensor(ReplicatedTensor):

    dispatch_id = 'private'

    def __init__(
        self,
        players,
        shares,
    ) -> None:
        # assert share0.shape.as_list() == share1.shape.as_list() == share2.shape.as_list()

        self.players = players
        self.shares = shares

    # @property
    # def shape(self) -> List[int]:
    #     return self.share0.shape.as_list()


def _private_input(prot, player, input_fn) -> ReplicatedPrivateTensor:
    pass


class Kernel:
    pass

class AddPrivatePrivate(Kernel):
    pass

Fixed10 = Fixed(10)
Fixed20 = Fixed(20)

fixed:
    - precision (interpretation)
    - backing (delegation)
    - public/private (representation)
    - players (delegation)
    - replicated/shared/.. (representation)

int:
    - backing (delegation)
    - public/private (representation)
    - players (property)
    - replicated/shared/..

Pond: fixed16, two_additive
BEDK19: quan8, three_replicated
ABY3: fixed12, three_replicated

# upscale
Convert<Fixed(p), Fixed(q)> forall q > p >= 0

# truncate
Convert<Fixed(p), Fixed(q)> forall 0 <= q < p

Add<X: Fixed(p), Y: Fixed(q)> :=
    Convert<Fixed(p), Fixed(q)>

Add<X: Fixed(p), Y: Fixed(p)> :=
    Add<Int, Int>

class Add


_add_private_private = AddPrivatePrivate(Fixed, Fixed)

class Tensor(abc):

    @property
    def extended_dtype(self):
        return (self.dtype, self.protection)

class Private(abc):
    pass

@enum.unique
class Protection(enum.Enum):
    
    # values that can be freely revealed to anyone
    PUBLIC = 1

    # values intended to be kept private
    PRIVATE = 2

    # values intended to only be shared in encrypted form
    ENCRYPTED = 3

class Public(Tensor): # can be revealed
    backing
    dtype: int64, fixed10
    protection: public

class ThreeReplicated(Tensor):

    @property
    def dtype(self):
        return self._dtype

    @property
    def dtype_extended(self):
        return (self.dtype, replicated_three)
    

    backing
    dtype: int64, fixed10
    protection: encrypted

class PrivateRing:
    pass

class AdditiveN:
    shares: Tuple[BackingTensor]

class Additive2:
    shares: Pair[BackingTensor]

class Replicated3:
    shares: Tuple[BackingTensor]

class PrivateInteger:
    element: PrivateRing

class PrivateFixedpoint<ER>:
    precision: int
    encoded: ER

class PrivateQuantized<ER>:
    precision: int
    zero_point: ER
    scale: ER

backing:
- ring32
- ring64
- ring100

encrypted:
- additive2
- additive2_masked
- replicated3

dtype:
- int32
- int64
- fixed16
- quant8

add<ring64, ring64>
add<ring64, int>
add<(additive2, ring32), (additive2, ring32)>
add<(additive2, ring64), (additive2, ring64)>
add<(additive2, ring32), (additive2, ring32)>

for backing in [..]:
    ops.put(add, (fixed16, additive2, backing), (fixed16, additive2, backing))

add<(fixed16, additive2, ring64), (fixed16, additive2, ring64)>
dot<(quant8, replicated3, ring64), (quant8, replicated3, ring64)>

add<PrivateInteger, PrivateFixedpoint>




class Additive(Tensor):
    backing
    dtype: int64, fixed10
    protection: encrypted


class Tensor

class PrivateTensor
class PublicTensor

class IntPrivateReplicated(Int64)
class IntPublicReplicated(Int64)
class FixedPrivateReplicated(Int64)
class FixedPublicReplicated(Int64)

class IntPrivateAdditive(Int64)
class IntPublicAdditive(Int64)
class FixedPrivateAdditive(Int64)
class FixedPublicAdditive(Int64)


int64 = IntPublic
pint64 


class Kernel:

    def __call__(self, *args):
        return self.run(*args)

class AddPrivatePrivate(Kernel):

    @property
    def signature(self):
        return 

    def __call__(self, x: ReplicatedPrivateTensor, y: ReplicatedPrivateTensor) -> ReplicatedPrivateTensor:
        players = x.players; assert y.players == players
        player0, player1, player2 = players
        
        x_on_0, x_on_1, x_on_2 = x.shares
        y_on_0, y_on_1, y_on_2 = y.shares

        with tf.device(player0.device_name):
            z1_on_0 = x_on_0[1] + y_on_0[1]
            z2_on_0 = x_on_0[2] + y_on_0[2]

        with tf.device(player1.device_name):
            z0_on_1 = x_on_1[0] + y_on_1[0]
            z2_on_1 = x_on_1[2] + y_on_1[2]

        with tf.device(player2.device_name):
            z0_on_2 = x_on_2[0] + y_on_2[0]
            z1_on_2 = x_on_2[1] + y_on_2[1]

        z_on_0 = (None, z1_on_0, z2_on_0)
        z_on_1 = (z0_on_1, None, z2_on_1)
        z_on_2 = (z0_on_2, z1_on_2, None)
        shares = (z_on_0, z_on_1, z_on_2)

        return ReplicatedPrivateTensor(players, shares)

add = Dispatcher(
    AddPrivatePrivate(),
    AddPrivatePublic(),
    AddPublicPrivate(),
    AddPublicPublic(),
)

class SubPrivatePrivate(Kernel):
    
    def __call__(self, x: ReplicatedPrivateTensor, y: ReplicatedPrivateTensor) -> ReplicatedPrivateTensor:
        players = x.players; assert y.players == players
        player0, player1, player2 = players
        
        x_on_0, x_on_1, x_on_2 = x.shares
        y_on_0, y_on_1, y_on_2 = y.shares

        with tf.device(player0.device_name):
            z1_on_0 = x_on_0[1] - y_on_0[1]
            z2_on_0 = x_on_0[2] - y_on_0[2]

        with tf.device(player1.device_name):
            z0_on_1 = x_on_1[0] - y_on_1[0]
            z2_on_1 = x_on_1[2] - y_on_1[2]

        with tf.device(player2.device_name):
            z0_on_2 = x_on_2[0] - y_on_2[0]
            z1_on_2 = x_on_2[1] - y_on_2[1]

        z_on_0 = (None, z1_on_0, z2_on_0)
        z_on_1 = (z0_on_1, None, z2_on_1)
        z_on_2 = (z0_on_2, z1_on_2, None)
        shares = (z_on_0, z_on_1, z_on_2)

        return ReplicatedPrivateTensor(players, shares)

sub = Dispatcher(
    SubPrivatePrivate(),
    SubPrivatePublic(),
    SubPublicPrivate(),
    SubPublicPublic(),
)

class MulPrivatePrivate(Kernel):

    def __call__(self, x: ReplicatedPrivateTensor, y: ReplicatedPrivateTensor) -> ReplicatedPrivateTensor:
        players = x.players; assert y.players == players
        player0, player1, player2 = players

        x_on_0, x_on_1, x_on_2 = x.shares
        y_on_0, y_on_1, y_on_2 = y.shares

        # get the next zero triple defined by seeds (s0, s1, s2) shared between the three players
        alpha0, alpha1, alpha2 = (5, 7, -12)

        with tf.device(player0.device_name):
            z2_on_0 = (x_on_0[2] * y_on_0[2] +
                       x_on_0[2] * y_on_0[1] +
                       x_on_0[1] * y_on_0[2])
            z2_on_1 = z2_on_0 + alpha0

        with tf.device(player1.device_name):
            z0_on_1 = (x_on_1[0] * y_on_1[0] +
                       x_on_1[0] * y_on_1[2] +
                       x_on_1[2] * y_on_1[0])
            z0_on_2 = z0_on_1 + alpha1

        with tf.device(player2.device_name):
            z1_on_2 = (x_on_2[1] * y_on_2[1] +
                       x_on_2[1] * y_on_2[0] +
                       x_on_2[0] * y_on_2[1])
            z1_on_0 = z1_on_2 + alpha2

        z0 = (None, z1_on_0, z2_on_0)
        z1 = (z0_on_1, None, z2_on_1)
        z2 = (z0_on_2, z1_on_2, None)
        shares = (z0, z1, z2)

        return ReplicatedPrivateTensor(players, shares)

mul = Dispatcher(
    MulPrivatePrivate(),
    MulPrivatePublic(),
    MulPublicPrivate(),
    MulPublicPublic(),
)
