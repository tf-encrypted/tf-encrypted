import unittest

import tensorflow as tf

# tf.enable_eager_execution()

import tf_encrypted as tfe
import numpy as np

from .context import Context
from .types import Dtypes
from .kernels import dispatch
from .replicated import zero_mask
from tf_encrypted.tensor import int64factory


class TestReplicated(unittest.TestCase):

    def test_zero_mask(self):
        tfe.set_config(tfe.LocalConfig([
            'server0', 'server1', 'server2'
        ]))

        players = tfe.get_config().get_players('server0, server1, server2')
        alpha0, alpha1, alpha2 = zero_mask(players, [5, 5], int64factory)

        out = alpha0 + alpha1 + alpha2

        np.testing.assert_equal(tfe.Session().run(out), np.zeros([5, 5]))

    def test_add(self):

        tfe.set_config(tfe.LocalConfig([
            'server0', 'server1', 'server2'
        ]))

        players = tfe.get_config().get_players('server0, server1, server2')

        input1 = tf.constant([3, 4, 5], tf.float32)
        input2 = tf.constant([3, 4, 5], tf.float32)

        context = Context()

        # these four calls should probably just be two tf.float32 -> REPLICATED3
        x = dispatch(context, "Cast", input1, Dtypes.FIXED10, players=None)
        y = dispatch(context, "Cast", input2, Dtypes.FIXED10, players=None)
        xs = dispatch(context, "Cast", x, Dtypes.REPLICATED3, players=players)
        ys = dispatch(context, "Cast", y, Dtypes.REPLICATED3, players=players)

        z = dispatch(context, "Add", xs, ys)

        # these two calls should probably be one REPLICATED3 -> tf.float32
        fixed = dispatch(context, "Cast", z, Dtypes.FIXED10, players=players)
        float = dispatch(context, "Cast", fixed, tf.float32, players=None)

        np.testing.assert_equal(tfe.Session().run(float), [6, 8, 10])

    def test_mul(self):

        tfe.set_config(tfe.LocalConfig([
            'server0', 'server1', 'server2'
        ]))

        players = tfe.get_config().get_players('server0, server1, server2')

        context = Context()

        input1 = tf.constant([5], tf.float32)
        input2 = tf.constant([5], tf.float32)

        # these four calls should probably just be two tf.float32 -> REPLICATED3
        x = dispatch(context, "Cast", input1, Dtypes.FIXED10, players=None)
        y = dispatch(context, "Cast", input2, Dtypes.FIXED10, players=None)
        xs = dispatch(context, "Cast", x, Dtypes.REPLICATED3, players=players)
        ys = dispatch(context, "Cast", y, Dtypes.REPLICATED3, players=players)

        z = dispatch(context, "Mul", xs, ys)

        # these two calls should probably be one REPLICATED3 -> tf.float32
        fixed = dispatch(context, "Cast", z, Dtypes.FIXED10, players=players)
        float = dispatch(context, "Cast", fixed, tf.float32, players=None)

        np.testing.assert_equal(tfe.Session().run(float).shape, [1])


if __name__ == '__main__':
    unittest.main()
