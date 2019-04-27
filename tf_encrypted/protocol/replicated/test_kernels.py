import unittest
import numpy as np

import tensorflow as tf

from .kernels import dispatch
from .types import Dtypes
from .context import Context
import tf_encrypted as tfe


class TestKernels(unittest.TestCase):
    def test_call_kernel(self):
        x = tf.constant([5], dtype=tf.float32)

        context = Context()

        out = dispatch(context, "Cast", x, Dtypes.FIXED10, players=None)

        np.testing.assert_array_equal(tfe.Session().run(out.backing), [5120])

    def test_call_kernel_with_attrs(self):
        x = tf.constant([5], dtype=tf.float32)

        context = Context()

        tfe.set_config(tfe.LocalConfig([
            'server0', 'server1', 'server2'
        ]))

        players = tfe.get_config().get_players('server0, server1, server2')

        out = dispatch(context, "Cast", x, Dtypes.FIXED10, players=players)

        np.testing.assert_array_equal(tfe.Session().run(out.backing), [5120])


if __name__ == '__main__':
    unittest.main()
