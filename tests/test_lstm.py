import unittest

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.contrib.rnn.python.ops import rnn_cell

from tensorflow_encrypted.layers import LSTM


class TestLstm(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_forward(self) -> None:

        input_shape = [3]
        x = np.array([[1., 1., 1.]])
        prev_c = 0.1 * np.asarray([[0, 1]])
        prev_h = 0.1 * np.asarray([[2, 3]])
        num_units = 2

        args = np.concatenate((prev_h, x), axis=1)
        out_size = 4 * num_units
        proj_size = args.shape[-1]
        weights = np.ones([proj_size, out_size]) * 0.5
        out = np.matmul(args, weights)
        bias = np.ones([out_size]) * 0.5
        concat = out + bias
        i, j, f, o = np.split(concat, 4, 1)

        config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto_producer'
        ])

        with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:

            i = prot.define_private_variable(i)
            j = prot.define_private_variable(j)
            o = prot.define_private_variable(o)
            f = prot.define_private_variable(f)
            prev_c = prot.define_private_variable(prev_c)

            lstm_layer = LSTM(input_shape, prev_c, f, i, o, j)
            lstm_out_pond = lstm_layer.forward()

            with config.session() as sess:
                sess.run(tf.global_variables_initializer())
                out_pond = lstm_out_pond.reveal().eval(sess)

            res = []

            # reset graph
            tf.reset_default_graph()

            with tf.Session() as sess:
              with variable_scope.variable_scope(
                  "other", initializer=init_ops.constant_initializer(0.5)) as vs:
                x = array_ops.zeros(
                  [1, 3])  # Test BasicLSTMCell with input_size != num_units.
                c = array_ops.zeros([1, 2])
                h = array_ops.zeros([1, 2])
                state = (c, h)
                cell = rnn_cell.LayerNormBasicLSTMCell(2, layer_norm=False)
                g, out_m = cell(x, state)
                sess.run([variables.global_variables_initializer()])
                res = sess.run([g, out_m], {
                  x.name: np.array([[1., 1., 1.]]),
                  c.name: 0.1 * np.asarray([[0, 1]]),
                  h.name: 0.1 * np.asarray([[2, 3]]),
                })

            np.testing.assert_array_almost_equal(out_pond, res[1].h, decimal=1)


if __name__ == '__main__':
    unittest.main()
