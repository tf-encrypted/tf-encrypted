import sys
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto_producer',
    'weights_provider',
    'prediction_client'
])

# config = tfe.RemoteConfig([
#     ('server0', 'localhost:4440'),
#     ('server1', 'localhost:4441'),
#     ('crypto_producer', 'localhost:4442'),
#     ('weights_provider', 'localhost:4443'),
#     ('prediction_client', 'localhost:4444')
# ])

if len(sys.argv) > 1:

    #
    # assume we're running as a server
    #

    player_name = str(sys.argv[1])

    server = config.server(player_name)
    server.start()
    server.join()

else:

    #
    # assume we're running as master
    #

    class WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            raw_w = np.array([5, 5, 5, 5]).reshape((2, 2))
            w = tf.constant(raw_w)
            return tf.Print(w, [w])

    class PredictionInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            x = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=tf.float32)
            return tf.Print(x, [x])

    class PredictionOutputReceiver(tfe.io.OutputReceiver):
        def receive_output(self, tensors):
            return tf.Print([], tensors, summarize=4)

    weights_input = WeightsInputProvider(config.get_player('weights_provider'))
    prediction_input = PredictionInputProvider(config.get_player('prediction_client'))
    prediction_output = PredictionOutputReceiver(config.get_player('prediction_client'))

    with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:

        # treat weights as private
        w, = prot.define_private_input(weights_input)

        # load input for prediction
        x, = prot.define_private_input(prediction_input)

        # compute prediction
        y = prot.dot(x, w)

        # send output
        prediction_op = prot.define_output([y], prediction_output)

        with config.session() as sess:
            tfe.run(sess, tf.global_variables_initializer(), tag='init')

            for _ in range(5):
                tfe.run(sess, prediction_op, tag='prediction')
