import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from .test_convert import export_matmul, read_graph
from tf_encrypted.private_model import PrivateModel


class TestArgMax(unittest.TestCase):
    def test_private_model(self):
        def provide_input():
            return tf.placeholder(dtype=tf.float32, shape=[1, 2], name="api/0")

        export_matmul("matmul.pb", [1, 2])

        graph_def = read_graph("matmul.pb")

        c = tfe.convert.convert.Converter()
        y = c.convert(graph_def, tfe.convert.register(), 'input-provider', provide_input)

        model = PrivateModel(y)

        output = model.predict(np.ones([1, 2]))

        np.testing.assert_array_equal(output, [[2.]])


if __name__ == '__main__':
    unittest.main()
