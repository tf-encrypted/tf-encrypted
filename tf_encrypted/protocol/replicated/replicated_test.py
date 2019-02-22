import unittest

import tensorflow as tf

import tf_encrypted as tfe
import numpy as np

from replicated import (
    ReplicatedPrivateTensor,
    AddPrivatePrivate,
    MulPrivatePrivate,
    zero_share,
    share,
    recombine,
)


class TestReplicated(unittest.TestCase):

    def test_zero_share(self):
        tfe.set_config(tfe.LocalConfig([
            'server0', 'server1', 'server2'
        ]))

        players = tfe.get_config().get_players('server0, server1, server2')
        alpha0, alpha1, alpha2 = zero_share(players, [5, 5])

        out = alpha0 + alpha1 + alpha2

        np.testing.assert_equal(tfe.Session().run(out), np.zeros([5, 5]))

    def test_add(self):

        tfe.set_config(tfe.LocalConfig([
            'server0', 'server1', 'server2'
        ]))

        players = tfe.get_config().get_players('server0, server1, server2')

        x = share(players, np.array([3, 4, 5]))
        y = share(players, np.array([3, 4, 5]))

        kern = AddPrivatePrivate()

        z = kern(x, y)

        final = recombine(players, z)

        np.testing.assert_equal(tfe.Session().run(final), [6, 8, 10])

    def test_mul(self):

        tfe.set_config(tfe.LocalConfig([
            'server0', 'server1', 'server2'
        ]))

        players = tfe.get_config().get_players('server0, server1, server2')

        x = share(players, np.array([3, 4, 5]))
        y = share(players, np.array([3, 4, 5]))

        kern = MulPrivatePrivate()
        z = kern(x, y)

        final = recombine(players, z)

        np.testing.assert_equal(tfe.Session().run(final), [9, 16, 25])


if __name__ == '__main__':
    unittest.main()
