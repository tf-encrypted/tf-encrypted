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


def share(v, sender, players):
    """ 
    Turn value `v` into a replicated sharing among `players`.

    Note that `sender` is assumed to currently hold `v`,
    and may be either a third party or one of the players.
    """

    # TODO(Morten) assume that `v` is a BackingTensor (RingTensor?)

    with tf.device(sender.device_name):

        # we need random tensors for two shares, which will (in most cases) be seeds
        r = v.backing_dtype.sample_uniform(v.shape)
        s = v.backing_dtype.sample_uniform(v.shape)

        # we form the third share, which will not be a seed
        t = v - r - s

        # logic below optimizes the distribution of seeds versus expanded tensors,
        # making sure that if the sender is one of the players then that player is
        # *not* the lucky one who only gets seeds (which would otherwise mean
        # that the expanded tensor is sent twice)
        if sender is players[0]:
            # this player will end up holding x1 and x2 so make sure one of those is t
            x0, x1, x2 = random.choice([
                (r, t, s),  # player1 is the lucky one
                (r, s, t),  # player2 is the lucky one
            ])
        elif sender is players[1]:
            # this player will end up holding x0 and x2 so make sure one of those is t
            x0, x1, x2 = random.choice([
                (t, r, s),  # player0 is the lucky one
                (r, s, t),  # player2 is the lucky one
            ])
        elif sender is players[2]:
            # this player will end up holding x0 and x1 so make sure one of those is t
            x0, x1, x2 = random.choice([
                (t, r, s),  # player0 is the lucky one
                (r, t, s),  # player1 is the lucky one
            ])
        else:
            # the sender is a third party so there are no constraints
            x0, x1, x2 = random.choice([
                (t, r, s),  # player0 is the lucky one
                (r, t, s),  # player1 is the lucky one
                (r, s, t),  # player2 is the lucky one
            ])

    shares = ((None, x1, x2),
              (x0, None, x2),
              (x0, x1, None))

    return ReplicatedPrivateTensor(players, shares)


def reconstruct(x: ReplicatedPrivateTensor, receiver):

    shares_on_0, shares_on_1, shares_on_2 = x.shares

    with tf.device(receiver.device_name):

        if receiver is x.players[0]:
            # we only need to have someone send x0
            x0, x1, x2 = random.choice([
                (shares_on_1[0], shares_on_0[1], shares_on_0[2]),  # player2 is the lucky one
                (shares_on_2[0], shares_on_0[1], shares_on_0[2]),  # player1 is the lucky one
            ])
        elif receiver is x.players[1]:
            # we only need to have someone send x1
            x0, x1, x2 = random.choice([
                (shares_on_1[0], shares_on_0[1], shares_on_1[2]),  # player2 is the lucky one
                (shares_on_1[0], shares_on_2[1], shares_on_1[2]),  # player0 is the lucky one
            ])
        elif receiver is x.players[2]:
            # we only need to have someone send x2
            x0, x1, x2 = random.choice([
                (shares_on_2[0], shares_on_2[1], shares_on_0[2]),  # player1 is the lucky one
                (shares_on_2[0], shares_on_2[1], shares_on_1[2]),  # player0 is the lucky one
            ])
        else:
            # we don't have anything, need two players to send shares

            # TODO(Morten)
            # we currently avoid having all three players send shares for batching purposes,
            # maybe this is sub-optmizal?

            x0, x1, x2 = random.choice([
                # player0 and player1
                (shares_on_1[0], shares_on_0[1], shares_on_0[2]),
                (shares_on_1[0], shares_on_0[1], shares_on_1[2]),
                # player0 and player2
                (shares_on_2[0], shares_on_0[1], shares_on_0[2]),
                (shares_on_2[0], shares_on_2[1], shares_on_0[2]),
                # player1 and player2
                (shares_on_1[0], shares_on_2[1], shares_on_1[2]),
                (shares_on_2[0], shares_on_2[1], shares_on_1[2]),
            ])

        v = x0 + x1 + x2
        return v


class ReplicatedTensor(abc.ABC):
    pass


class ReplicatedPrivateTensor(ReplicatedTensor):

    def __init__(self, players, shares):
        self.players = players
        self.shares = shares
        # TODO(Morten) assert that all non-None have same shape

    @property
    def shape(self):
        return self.shares[0][1].shape

    @property
    def backing_dtype(self):
        return self.shares[0][1].backing_dtype


class Kernel:
    def __call__(self, *args):
        pass


class AddPrivatePrivate(Kernel):

    def __call__(self, x: ReplicatedPrivateTensor, y: ReplicatedPrivateTensor) -> ReplicatedPrivateTensor:
        assert x.players == y.players
        assert x.backing_dtype == y.backing_dtype

        players = x.players
        x_on_0, x_on_1, x_on_2 = x.shares
        y_on_0, y_on_1, y_on_2 = y.shares

        with tf.device(players[0].device_name):
            z1_on_0 = x_on_0[1] + y_on_0[1]
            z2_on_0 = x_on_0[2] + y_on_0[2]

        with tf.device(players[1].device_name):
            z0_on_1 = x_on_1[0] + y_on_1[0]
            z2_on_1 = x_on_1[2] + y_on_1[2]

        with tf.device(players[2].device_name):
            z0_on_2 = x_on_2[0] + y_on_2[0]
            z1_on_2 = x_on_2[1] + y_on_2[1]

        z_on_0 = (None, z1_on_0, z2_on_0)
        z_on_1 = (z0_on_1, None, z2_on_1)
        z_on_2 = (z0_on_2, z1_on_2, None)
        shares = (z_on_0, z_on_1, z_on_2)

        return ReplicatedPrivateTensor(players, shares)


class SubPrivatePrivate(Kernel):

    def __call__(self, x: ReplicatedPrivateTensor, y: ReplicatedPrivateTensor) -> ReplicatedPrivateTensor:
        assert x.players == y.players
        assert x.backing_dtype == y.backing_dtype

        players = x.players
        x_on_0, x_on_1, x_on_2 = x.shares
        y_on_0, y_on_1, y_on_2 = y.shares

        with tf.device(players[0].device_name):
            z1_on_0 = x_on_0[1] - y_on_0[1]
            z2_on_0 = x_on_0[2] - y_on_0[2]

        with tf.device(players[1].device_name):
            z0_on_1 = x_on_1[0] - y_on_1[0]
            z2_on_1 = x_on_1[2] - y_on_1[2]

        with tf.device(players[2].device_name):
            z0_on_2 = x_on_2[0] - y_on_2[0]
            z1_on_2 = x_on_2[1] - y_on_2[1]

        z_on_0 = (None, z1_on_0, z2_on_0)
        z_on_1 = (z0_on_1, None, z2_on_1)
        z_on_2 = (z0_on_2, z1_on_2, None)
        shares = (z_on_0, z_on_1, z_on_2)

        return ReplicatedPrivateTensor(players, shares)


class MulPrivatePrivate(Kernel):

    def __call__(self, x: ReplicatedPrivateTensor, y: ReplicatedPrivateTensor) -> ReplicatedPrivateTensor:
        assert x.players == y.players
        assert x.backing_dtype == y.backing_dtype
        
        players = x.players
        x_on_0, x_on_1, x_on_2 = x.shares
        y_on_0, y_on_1, y_on_2 = y.shares

        with tf.device(players[0].device_name):
            z2_on_0 = (x_on_0[2] * y_on_0[2]
                       + x_on_0[2] * y_on_0[1]
                       + x_on_0[1] * y_on_0[2])

        with tf.device(players[1].device_name):
            z0_on_1 = (x_on_1[0] * y_on_1[0]
                       + x_on_1[0] * y_on_1[2]
                       + x_on_1[2] * y_on_1[0])

        with tf.device(players[2].device_name):
            z1_on_2 = (x_on_2[1] * y_on_2[1]
                       + x_on_2[1] * y_on_2[0]
                       + x_on_2[0] * y_on_2[1])

        # get the next zero mask shared between the players
        alpha0, alpha1, alpha2 = zero_mask(
            players,
            shape=z2_on_0.shape,
            backing_dtype=z2_on_0.backing_dtype,
        )

        with tf.device(players[0].device_name):
            z2_on_1 = z2_on_0 + alpha0

        with tf.device(players[1].device_name):
            z0_on_2 = z0_on_1 + alpha1

        with tf.device(players[2].device_name):
            z1_on_0 = z1_on_2 + alpha2

        z0 = (None, z1_on_0, z2_on_0)
        z1 = (z0_on_1, None, z2_on_1)
        z2 = (z0_on_2, z1_on_2, None)
        shares = (z0, z1, z2)

        return ReplicatedPrivateTensor(players, shares)
