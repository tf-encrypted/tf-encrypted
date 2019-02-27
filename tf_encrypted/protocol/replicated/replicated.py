import abc
import tensorflow as tf
from tf_encrypted.operations.secure_random import seed, seeded_random_uniform
from tf_encrypted.tensor.helpers import inverse
import numpy as np

maxval = 9223372036854775807
minval = -9223372036854775808
dtype = tf.int64
modulus = 18446744073709551616
base = 2
precision_fractional = 10


def encode(value):
    return tf.cast(value * (base ** precision_fractional), tf.int64)


def truncate(players, value):
    player0, player1, player2 = players

    x_on_0, x_on_1, x_on_2 = value.shares

    with tf.device(player0.device_name):
        s = seed()
        r0 = seeded_random_uniform(value.shape, dtype=dtype, seed=s, minval=minval, maxval=maxval)
        z2_on_0 = r0
        z1_on_0 = tf.bitwise.right_shift(x_on_0[1] + x_on_0[2], precision_fractional) - r0

    with tf.device(player1.device_name):
        r1 = seeded_random_uniform(value.shape, dtype=dtype, seed=s, minval=minval, maxval=maxval)
        z2_on_1 = r1
        z0_on_1 = tf.bitwise.right_shift(x_on_1[0], precision_fractional)

    with tf.device(player2.device_name):
        z0_on_2 = tf.bitwise.right_shift(x_on_2[0], precision_fractional)
        z1_on_2 = z1_on_0

    z_on_0 = (None, z1_on_0, z2_on_0)
    z_on_1 = (z0_on_1, None, z2_on_1)
    z_on_2 = (z0_on_2, z1_on_2, None)
    shares = (z_on_0, z_on_1, z_on_2)

    return ReplicatedPrivateTensor(players, shares)


def zero_share(players, shape, backing_dtype):

    with tf.device(players[0].device_name):
        r0 = backing_dtype.sample_uniform(shape)

    with tf.device(players[1].device_name):
        r1 = backing_dtype.sample_uniform(shape)

    with tf.device(players[2].device_name):
        r2 = backing_dtype.sample_uniform(shape)

    with tf.device(players[0].device_name):
        alpha0 = r0 - r1

    with tf.device(players[1].device_name):
        alpha1 = r1 - r2
    
    with tf.device(players[2].device_name):
        alpha2 = r2 - r0

    return alpha0, alpha1, alpha2


def share(players, x):
    alpha0, alpha1, alpha2 = zero_share(players, x.shape)

    x1 = alpha1
    x2 = alpha2

    with tf.device(players[0].device_name):
        x0 = x + alpha0

    xs = ReplicatedPrivateTensor(players, ((None, x1, x2), (x0, None, x2), (x0, x1, None)))

    return xs


def recombine(players, z):
    with tf.device(players[0].device_name):
        final = z.shares[2][0] + z.shares[0][1] + z.shares[1][2]

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
            z2_on_0 = (x_on_0[2] * y_on_0[2]
                       + x_on_0[2] * y_on_0[1]
                       + x_on_0[1] * y_on_0[2])
            z2_on_1 = z2_on_0 + alpha0

        with tf.device(player1.device_name):
            z0_on_1 = (x_on_1[0] * y_on_1[0]
                       + x_on_1[0] * y_on_1[2]
                       + x_on_1[2] * y_on_1[0])
            z0_on_2 = z0_on_1 + alpha1

        with tf.device(player2.device_name):
            z1_on_2 = (x_on_2[1] * y_on_2[1]
                       + x_on_2[1] * y_on_2[0]
                       + x_on_2[0] * y_on_2[1])
            z1_on_0 = z1_on_2 + alpha2

        z0 = (None, z1_on_0, z2_on_0)
        z1 = (z0_on_1, None, z2_on_1)
        z2 = (z0_on_2, z1_on_2, None)
        shares = (z0, z1, z2)

        return ReplicatedPrivateTensor(players, shares)
