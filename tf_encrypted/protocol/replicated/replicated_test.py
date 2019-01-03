import unittest

import tf_encrypted as tfe
import numpy as np

from .replicated import Replicated, ReplicatedPrivateTensor, _add_private_private


class TestReplicated(unittest.TestCase):

    def test_add(self):

        tfe.set_config(tfe.LocalConfig([
            'server0', 'server1', 'server2'
        ]))

        prot = Replicated()

        x = ReplicatedPrivateTensor(prot, (1,2), (0,2), (0,1))
        y = ReplicatedPrivateTensor(prot, (1,2), (0,2), (0,1))

        z = _add_private_private(prot, x, y)

        assert z.shares0 == (1+1, 2+2)
        assert z.shares1 == (0+0, 2+2)
        assert z.shares2 == (0+0, 1+1)

    def test_mul(self):

        tfe.set_config(tfe.LocalConfig([
            'server0', 'server1', 'server2'
        ]))

        prot = Replicated()

        x = ReplicatedPrivateTensor(prot, (1,2), (0,2), (0,1))
        y = 10

        z = _mul_private_public(prot, x, y)

        assert z.shares0 == (10, 20)
        assert z.shares1 == (0+0, 2+2)
        assert z.shares2 == (0+0, 1+1)


if __name__ == '__main__':
    unittest.main()
