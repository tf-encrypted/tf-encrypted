# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import tf_encrypted as tfe
from tf_encrypted.private_model import PrivateModel
from tf_encrypted.private_model import secure_model
# TODO this is a bit weird
from tf_encrypted.convert.convert_test import read_graph, export_matmul


class TestPrivateModel(unittest.TestCase):
  def test_private_model(self):
    def provide_input():
      return tf.placeholder(dtype=tf.float32, shape=[1, 2], name="api/0")

    export_matmul("matmul.pb", [1, 2])

    graph_def = read_graph("matmul.pb")

    with tfe.protocol.Pond():
      c = tfe.convert.convert.Converter(tfe.convert.registry())
      y = c.convert(graph_def, 'input-provider', provide_input)

      model = PrivateModel(y)

      output = model.private_predict(np.ones([1, 2]))

    np.testing.assert_array_equal(output, [[2.]])


class TestSecureModel(unittest.TestCase):

  def setUp(self):
    K.clear_session()

  def tearDown(self):
    K.clear_session()

  def test_secure_model(self):
    with tfe.protocol.Pond():
      tf.random.set_random_seed(42)

      d = tf.keras.layers.Dense(1, input_shape=(10,), use_bias=False)
      model = tf.keras.Sequential([
          d
      ])

      x = np.ones((1, 10))
      y = model.predict(x)

      s_model = secure_model(model)
      s_y = s_model.private_predict(x)

      np.testing.assert_array_almost_equal(s_y, y, 4)

  def test_secure_model_batch(self):
    with tfe.protocol.Pond():
      tf.random.set_random_seed(42)

      d = tf.keras.layers.Dense(1, input_shape=(10,), use_bias=False)
      model = tf.keras.Sequential([
          d
      ])

      x = np.ones((2, 10))
      y = model.predict(x)

      s_model = secure_model(model, batch_size=2)
      s_y = s_model.private_predict(x)

      np.testing.assert_array_almost_equal(s_y, y, 4)

if __name__ == '__main__':
  unittest.main()
