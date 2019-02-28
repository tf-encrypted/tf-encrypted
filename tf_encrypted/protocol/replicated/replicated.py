import random

import tensorflow as tf

from tf_encrypted.operations.secure_random import seed, seeded_random_uniform
from .ops import Add, Mul, Sub, Cast
from .types import Dtypes, Fixed, fixed_config

maxval = 9223372036854775807
minval = -9223372036854775808
dtype = tf.int64
modulus = 18446744073709551616


def encode(value, base, exp):
    return value * (base ** exp)


def decode(value, base, exp):
    return value / (base ** exp)


def truncate(value):
    player0, player1, player2 = value.players

    x_on_0, x_on_1, x_on_2 = value.shares
    precision_fractional = x_on_0[1].precision_fractional

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

    return ReplicatedPrivate(value.players, shares)


def zero_mask(players, shape, backing_dtype):

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

    return ReplicatedPrivate(players, shares)


def reconstruct(x: ReplicatedPrivate, receiver):

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


class Replicated(abc.ABC):
    pass


class ReplicatedPrivate(Replicated):

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


    return ReplicatedPrivate(players, shares)


# TODO Subclass ProtocolTensor?
class ReplicatedPrivate():
    def __init__(self, players, shares):
        self.players = players
        self.shares = shares

        for share in self.shares[0]:
            if share is not None:
                self._shape = share.shape
                self.base_dtype = share.dtype
                break

    @property
    def dtype(self):
        return (Dtypes.REPLICATED3, self.base_dtype)

    @property
    def shape(self):
        return self._shape

    @property
    def backing(self):
        return self.shares


# TODO benefits of a bass class??
class Kernel:
    def __call__(self, *args):
        pass


class AddPrivatePrivate(Kernel):
    op = Add

    def __call__(self, x: ReplicatedPrivate, y: ReplicatedPrivate) -> ReplicatedPrivate:
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

        return ReplicatedPrivate(players, shares)


class SubPrivatePrivate(Kernel):
    op = Sub

    def __call__(self, x: ReplicatedPrivate, y: ReplicatedPrivate) -> ReplicatedPrivate:
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

        return ReplicatedPrivate(players, shares)


class MulPrivatePrivate(Kernel):
    op = Mul

    def __call__(self, x: ReplicatedPrivate, y: ReplicatedPrivate) -> ReplicatedPrivate:
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

        # if x.base_dtype is Dtypes.FIXED10 or x.base_dtype is Dtypes.FIXED16:
        #     return truncate(ReplicatedPrivate(players, shares))

        return ReplicatedPrivate(players, shares)


class CastIntReplicated3(Kernel):
    op = Cast

    def __call__(self, context, value, dtype, players=None):
        return share(players, value)


class CastFloat32Fixed(Kernel):
    op = Cast

    def __call__(self, context, value, dtype, players=None):
        base = fixed_config[dtype]["base"]
        bits = fixed_config[dtype]["bits"]

        fixed = tf.cast(encode(value, base, bits), tf.int64)

        return Fixed(fixed, dtype)


class CastFixedFloat32(Kernel):
    op = Cast

    def __call__(self, context, value, dtype, players=None):
        base = fixed_config[value.dtype]["base"]
        bits = fixed_config[value.dtype]["bits"]

        return decode(value.backing, base, bits)


class CastReplicated3Int(Kernel):
    op = Cast

    def __call__(self, context, value, dtype, players=None):
        return reconstruct(players, value)
