# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.engine.input_layer import Input

np.random.seed(42)


class TestInput(unittest.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def test_input(self):
    x = Input(shape=(2,), batch_size=1)
    fd = x.feed(np.random.normal(size=(1, 2)))

    with tfe.Session() as sess:
      sess.run(x.reveal(), feed_dict=fd)


if __name__ == '__main__':
  unittest.main()
