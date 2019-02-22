import sys

import tensorflow as tf
import tf_encrypted as tfe


# use configuration from file if specified (otherwise fall back to default LocalConfig)
if len(sys.argv) >= 2:
    # config file was specified
    config_file = sys.argv[1]
    config = tfe.RemoteConfig.load(config_file)
    tfe.set_config(config)
    tfe.set_protocol(tfe.protocol.Pond())


def provide_input() -> tf.Tensor:
    # pick random tensor to be averaged
    return tf.random_normal(shape=(10,))


# get input from inputters as private values
inputs = [
    tfe.define_private_input('inputter-0', provide_input),
    tfe.define_private_input('inputter-1', provide_input),
    tfe.define_private_input('inputter-2', provide_input),
    tfe.define_private_input('inputter-3', provide_input),
    tfe.define_private_input('inputter-4', provide_input),
]


# sum all inputs and divide by count
result = tfe.add_n(inputs) / len(inputs)


def receive_output(average: tf.Tensor) -> tf.Operation:
    # simply print average
    return tf.print("Average:", average)


# send result to receiver
result_op = tfe.define_output('result-receiver', result, receive_output)


# run a few times
with tfe.Session() as sess:
    sess.run(result_op, tag='average')
