import unittest

import tf_encrypted as tfe
import numpy as np

from .replicated import (
    ReplicatedPrivateTensor,
    _add_private_private,
    _mul_private_private,
)


class TestReplicated(unittest.TestCase):

    def test_add(self):

        tfe.set_config(tfe.LocalConfig([
            'server0', 'server1', 'server2'
        ]))

        players = tfe.get_config().get_players('server0, server1, server2')

        x = ReplicatedPrivateTensor(players, ((None,1,2), (0,None,2), (0,1,None)))
        y = ReplicatedPrivateTensor(players, ((None,1,2), (0,None,2), (0,1,None)))

        z = _add_private_private(x, y)

        assert z.shares[0][1] + z.shares[1][2] + z.shares[2][0] == 6
        assert z.shares[0][0] is None
        assert z.shares[1][1] is None
        assert z.shares[2][2] is None

    def test_mul(self):

        tfe.set_config(tfe.LocalConfig([
            'server0', 'server1', 'server2'
        ]))

        players = tfe.get_config().get_players('server0, server1, server2')

        x = ReplicatedPrivateTensor(players, ((None,1,2), (0,None,2), (0,1,None)))
        y = ReplicatedPrivateTensor(players, ((None,1,2), (0,None,2), (0,1,None)))

        z = _mul_private_private(x, y)

        assert z.shares[0][1] + z.shares[1][2] + z.shares[2][0] == 9
        assert z.shares[0][0] is None
        assert z.shares[1][1] is None
        assert z.shares[2][2] is None


if __name__ == '__main__':
    unittest.main()
