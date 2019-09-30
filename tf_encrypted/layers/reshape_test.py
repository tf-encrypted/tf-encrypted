# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.layers import Reshape
from tf_encrypted.utils import unwrap_fetches

def test_forward():
  tf.compat.v1.enable_v2_behavior()

  input_shape = [2, 3, 4, 5]
  output_shape = [2, -1]
  input_reshape = np.random.standard_normal(input_shape)

  # reshape pond
  with tfe.protocol.Pond() as prot:
    @tf.function
    def pond():
      reshape_input = prot.define_private_tensor(input_reshape)
      reshape_layer = Reshape(input_shape, output_shape)

      reshape_out_pond = reshape_layer.forward(reshape_input)

      return unwrap_fetches(reshape_out_pond.reveal())

    out_pond = pond()

    x = tf.constant(input_reshape, dtype=tf.float32)

    out_tensorflow = tf.reshape(x, output_shape)

    assert np.isclose(out_pond, out_tensorflow, atol=0.6).all()


if __name__ == '__main__':
  unittest.main()
