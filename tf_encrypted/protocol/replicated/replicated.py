from typing import Optional, Tuple, List
import abc
import sys
import math
import enum

import tensorflow as tf

from tf_encrypted.operations.secure_random import seed, seeded_random_uniform


#
#
# class ReplicatedPublicTensor(ReplicatedTensor):
#     pass
#
#

#
#
# class Tensor(abc.ABC):
#
#     @property
#     def extended_dtype(self):
#         return (self.dtype, self.protection)

# class ThreeReplicated(Tensor):
#
#     @property
#     def dtype(self):
#         return self._dtype
#
#     @property
#     def dtype_extended(self):
#         return (self.dtype, replicated_three)

@enum.unique
class Protection(enum.Enum):

    # values that can be freely revealed to anyone
    PUBLIC = 1

    # values intended to be kept private
    PRIVATE = 2

    # values intended to only be shared in encrypted form
    ENCRYPTED = 3


maxval = 100


def zero_share(players, shape):
    player1, player2, player3 = players

    with tf.device(player3.device_name):
        s2 = seed()

    with tf.device(player2.device_name):
        s1 = seed()
        alpha1 = seeded_random_uniform(shape, seed=s1, maxval=maxval) - seeded_random_uniform(shape, seed=s2, maxval=maxval)

    with tf.device(player1.device_name):
        s0 = seed()
        alpha0 = seeded_random_uniform(shape, seed=s0, maxval=maxval) - seeded_random_uniform(shape, seed=s1, maxval=maxval)

    with tf.device(player3.device_name):
        alpha2 = seeded_random_uniform(shape, seed=s2, maxval=maxval) - seeded_random_uniform(shape, seed=s0, maxval=maxval)

    return alpha0, alpha1, alpha2


def share(players, x):
    alpha0, alpha1, alpha2 = zero_share(players, x.shape)

    x1 = alpha1
    x2 = alpha2

    with tf.device(players[0].device_name):
        x0 = x + alpha0

    xs = ReplicatedPrivateTensor(players, ((None, x1, x2), (x0, None, x2), (x0, x2, None)))

    return xs


def recombine(players, z):
    with tf.device(players[0].device_name):
        final = z.shares[0][1] + z.shares[1][2] + z.shares[2][0]

    return final


class ReplicatedTensor(abc.ABC):
    pass


class ReplicatedPrivateTensor(ReplicatedTensor):
    def __init__(
        self,
        players,
        shares,
    ) -> None:
        self.players = players
        self.shares = shares

    @property
    def shape(self):
        # cache shape?!
        for share in self.shares[0]:
            if share is not None:
                return share[0].shape


class Kernel:
    def __call__(self, *args):
        pass


class AddPrivatePrivate(Kernel):
    def __call__(self, x: ReplicatedPrivateTensor, y: ReplicatedPrivateTensor) -> ReplicatedPrivateTensor:
        players = x.players
        assert y.players == players
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


class SubPrivatePrivate(Kernel):

    def __call__(self, x: ReplicatedPrivateTensor, y: ReplicatedPrivateTensor) -> ReplicatedPrivateTensor:
        players = x.players
        assert y.players == players
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


class MulPrivatePrivate(Kernel):

    def __call__(self, x: ReplicatedPrivateTensor, y: ReplicatedPrivateTensor) -> ReplicatedPrivateTensor:
        players = x.players
        assert y.players == players
        player0, player1, player2 = players

        x_on_0, x_on_1, x_on_2 = x.shares
        y_on_0, y_on_1, y_on_2 = y.shares

        # get the next zero triple defined by seeds (s0, s1, s2) shared between the three players
        alpha0, alpha1, alpha2 = zero_share(players, x.shape)

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
