from typing import List
import sys
from functools import reduce

import tensorflow as tf
import tensorflow_encrypted as tfe


player_names_fixed = ['server0', 'server1', 'crypto_producer', 'result_receiver']

if len(sys.argv) >= 2:
    # config file was specified
    config_file = sys.argv[1]
    config = tfe.config.load(config_file)
    player_names_inputter = [player.name for player in config.players if player.name not in player_names_fixed]

else:
    # create a local config with all inputters and the usual suspects needed for the Pond protocol
    player_names_inputter = ['inputter-{}'.format(i) for i in range(5)]
    config = tfe.LocalConfig(player_names_fixed + player_names_inputter)


class Inputter(tfe.io.InputProvider):
    def provide_input(self) -> List[tf.Tensor]:
        # pick random tensor to be averaged
        return [tf.random_normal(shape=(10,))]


class ResultReceiver(tfe.io.OutputReceiver):
    def receive_output(self, tensors: List[tf.Tensor]) -> tf.Operation:
        average, = tensors
        # simply print average
        return tf.Print([], [average], summarize=10, message="Average:")


# create players based on names from above
inputters = [Inputter(config.get_player(name)) for name in player_names_inputter]
result_receiver = ResultReceiver(config.get_player('result_receiver'))
server0 = config.get_player('server0')
server1 = config.get_player('server1')
crypto_producer = config.get_player('crypto_producer')

with tfe.protocol.Pond(server0, server1, crypto_producer) as prot:

    # get input from inputters as private values
    inputs = [prot.define_private_input(inputter)[0] for inputter in inputters]

    # sum all inputs and divide by count
    result = reduce(lambda x, y: x + y, inputs) * (1 / len(inputs))

    # send result to receiver
    result_op = prot.define_output([result], result_receiver)

    with tfe.Session() as sess:
        for _ in range(3):
            sess.run(result_op, tag='average')
