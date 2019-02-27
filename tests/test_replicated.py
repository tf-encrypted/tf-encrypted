import unittest

import tensorflow as tf

# tf.enable_eager_execution()

import tf_encrypted as tfe
import numpy as np

from tf_encrypted.protocol.replicated import (
    AddPrivatePrivate,
    MulPrivatePrivate,
    zero_share,
    share,
    recombine,
    truncate,
    dispatch,
    Context,
    Dtypes
)


class TestReplicated(unittest.TestCase):

    # def test_truncate(self):
    #     tfe.set_config(tfe.LocalConfig([
    #         'server0', 'server1', 'server2'
    #     ]))
    #
    #     players = tfe.get_config().get_players('server0, server1, server2')
    #
    #     x = encode(tf.constant([5], dtype=tf.float32))
    #     y = encode(tf.constant([5], dtype=tf.float32))
    #     xs = share(players, x)
    #     ys = share(players, y)
    #
    #     kern = MulPrivatePrivate()
    #     zs = kern(xs, ys)
    #
    #     t = truncate(players, zs)
    #
    #     final = recombine(players, t)
    #
    #     print(final / 2 ** 10)

    # def test_zero_share(self):
    #     tfe.set_config(tfe.LocalConfig([
    #         'server0', 'server1', 'server2'
    #     ]))
    #
    #     players = tfe.get_config().get_players('server0, server1, server2')
    #     alpha0, alpha1, alpha2 = zero_share(players, [5, 5])
    #
    #     out = alpha0 + alpha1 + alpha2
    #
    #     np.testing.assert_equal(tfe.Session().run(out), np.zeros([5, 5]))
    #
    # def test_add(self):
    #
    #     tfe.set_config(tfe.LocalConfig([
    #         'server0', 'server1', 'server2'
    #     ]))
    #
    #     players = tfe.get_config().get_players('server0, server1, server2')
    #
    #     x = share(players, np.array([3, 4, 5]))
    #     y = share(players, np.array([3, 4, 5]))
    #
    #     kern = AddPrivatePrivate()
    #
    #     z = kern(x, y)
    #
    #     final = recombine(players, z)
    #
    #     np.testing.assert_equal(tfe.Session().run(final), [6, 8, 10])
    #

    def test_mul(self):

        tfe.set_config(tfe.LocalConfig([
            'server0', 'server1', 'server2'
        ]))

        players = tfe.get_config().get_players('server0, server1, server2')

        context = Context()

        input1 = tf.constant([5], tf.float32)
        input2 = tf.constant([5], tf.float32)

        x = dispatch(context, "Cast", input1, Dtypes.FIXED10, players=players)
        y = dispatch(context, "Cast", input2, Dtypes.FIXED10, players=players)

        xs = dispatch(context, "Cast", x, Dtypes.REPLICATED3, players=players)
        ys = dispatch(context, "Cast", y, Dtypes.REPLICATED3, players=players)

        z = dispatch(context, "Mul", xs, ys)

        fixed = dispatch(context, "Cast", z, Dtypes.FIXED10, players=None)
        float = dispatch(context, "Cast", fixed, Dtypes.FIXED10, players=None)

        np.testing.assert_equal(final.numpy(), [9, 16, 25])


if __name__ == '__main__':
    unittest.main()
