from typing import List
import sys
from functools import reduce

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

NUM_INPUTS = 5

# give names to all inputters so we can create players for them below
inputter_names = ['inputter-{}'.format(i) for i in range(NUM_INPUTS)]
# create a config with all inputters and the usual suspects needed for the Pond protocol
config = tfe.LocalConfig(['server0', 'server1', 'crypto_producer', 'result_receiver'] + inputter_names)

if len(sys.argv) > 1:

    ####################################
    # assume we're running as a server #
    ####################################

    player_name = str(sys.argv[1])

    # pylint: disable=E1101
    server = config.server(player_name)
    server.start()
    server.join()

else:

    ##################################
    # assume we're running as master #
    ##################################

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
    inputters = [Inputter(config.get_player(name)) for name in inputter_names]
    result_receiver = ResultReceiver(config.get_player('result_receiver'))
    server0 = config.get_player('server0')
    server1 = config.get_player('server1')
    crypto_producer = config.get_player('crypto_producer')

    with tfe.protocol.Pond(server0, server1, crypto_producer) as prot:

        # get input from inputters as private values
        inputs = [prot.define_private_input(inputter)[0] for inputter in inputters]

        # sum all inputs and divide by count
        result = reduce(lambda x,y: x+y, inputs) * (1/len(inputs))

        # send result to receiver
        result_op = prot.define_output([result], result_receiver)

        with config.session() as sess:
            tfe.run(sess, tf.global_variables_initializer())            
            for _ in range(3):
                tfe.run(sess, result_op, tag='average')
