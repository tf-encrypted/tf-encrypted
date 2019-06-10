# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

# TODO this is a bit weird
from ..convert.convert_test import run_pad


class TestPad(unittest.TestCase):
  def setUp(self):
    tf.reset_default_graph()

  def test_pad(self):

    with tfe.protocol.Pond() as prot:

      tf.reset_default_graph()

      x_in = np.array([[1, 2, 3], [4, 5, 6]])
      input_input = prot.define_private_variable(x_in)

      paddings = [[2, 2], [3, 4]]

      out = prot.pad(input_input, paddings)

      with tfe.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out_tfe = sess.run(out.reveal())

      tf.reset_default_graph()

      out_tensorflow = run_pad(x_in)

      np.testing.assert_allclose(out_tfe, out_tensorflow, atol=.01)


if __name__ == '__main__':
  unittest.main()
