"""Example of a simple average using TF Encrypted."""

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

@tfe.local_computation
def provide_input() -> tf.Tensor:
  # pick random tensor to be averaged
  return tf.random_normal(shape=(10,))

@tfe.local_computation('result-receiver')
def receive_output(average: tf.Tensor) -> tf.Operation:
  # simply print average
  return tf.print("Average:", average)


if __name__ == '__main__':
  # get input from inputters as private values
  inputs = [provide_input(player_name="inputter-{}".format(i))  # pylint: disable=unexpected-keyword-arg
            for i in range(5)]

  # sum all inputs and divide by count
  result = tfe.add_n(inputs) / len(inputs)

  # send result to receiver
  result_op = receive_output(result)

  # run a few times
  with tfe.Session() as sess:
    sess.run(result_op, tag='average')
