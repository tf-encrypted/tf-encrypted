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


def receive_output(average: tf.Tensor) -> tf.Operation:
    # simply print average
    return tf.print("Average:", average)


# get input from inputters as private values
inputs = [
    tfe.define_private_input(
        'inputter-{}'.format(index),
        provide_input
    )
    for index in range(5)
]

# sum all inputs and divide by count
result = tfe.add_n(inputs) / len(inputs)

# send result to receiver
result_op = tfe.define_output('result-receiver', result, receive_output)

# run a few times
with tfe.Session() as sess:
    for _ in range(3):
        sess.run(result_op, tag='average')
