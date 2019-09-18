"""Example of a simple average using TF Encrypted."""

import logging
import sys

import tensorflow as tf
import tf_encrypted as tfe

# use configuration from file if specified
# otherwise, fall back to default LocalConfig
if len(sys.argv) >= 2:
  # config file was specified
  config_file = sys.argv[1]
  config = tfe.RemoteConfig.load(config_file)
  tfe.set_config(config)
  tfe.set_protocol(tfe.protocol.Pond())

@tfe.local_computation(name_scope='provide_input')
def provide_input() -> tf.Tensor:
  # pick random tensor to be averaged
  return tf.random_normal(shape=(10,))

@tfe.local_computation('result-receiver', name_scope='receive_output')
def receive_output(average: tf.Tensor) -> tf.Operation:
  # simply print average
  return tf.print("Average:", average)


if __name__ == '__main__':

  logging.basicConfig(level=logging.DEBUG)

  # get input from inputters as private values
  inputs = [
      provide_input(player_name='inputter-0'),  # pylint: disable=unexpected-keyword-arg
      provide_input(player_name='inputter-1'),  # pylint: disable=unexpected-keyword-arg
      provide_input(player_name='inputter-2'),  # pylint: disable=unexpected-keyword-arg
      provide_input(player_name='inputter-3'),  # pylint: disable=unexpected-keyword-arg
      provide_input(player_name='inputter-4'),  # pylint: disable=unexpected-keyword-arg
  ]

  # sum all inputs and divide by count
  result = tfe.add_n(inputs) / len(inputs)

  # send result to receiver
  result_op = receive_output(result)

  # run a few times
  with tfe.Session() as sess:
    sess.run(result_op, tag='average')
