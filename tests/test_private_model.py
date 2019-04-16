import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.private_model import PrivateModel
from .test_convert import read_graph, export_matmul


class TestPrivateModel(unittest.TestCase):
    def test_private_model(self):
        def provide_input():
            return tf.placeholder(dtype=tf.float32, shape=[1, 2], name="api/0")

        export_matmul("matmul.pb", [1, 2])

        graph_def = read_graph("matmul.pb")

        with tfe.protocol.Pond():
            c = tfe.convert.convert.Converter()
            y = c.convert(graph_def, tfe.convert.registry(), 'input-provider', provide_input)

            model = PrivateModel(y)

            output = model.private_predict(np.ones([1, 2]))

        np.testing.assert_array_equal(output, [[2.]])


if __name__ == '__main__':
    unittest.main()
