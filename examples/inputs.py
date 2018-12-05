import sys

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

config = tfe.get_config()

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

    def provide_weights() -> tf.Tensor:
        raw_w = np.array([5, 5, 5, 5]).reshape((2, 2))
        w = tf.constant(raw_w)
        return tf.print(w, [w])

    def provide_input() -> tf.Tensor:
        x = tf.constant([1, 2, 3, 4], shape=(2, 2), dtype=tf.float32)
        return tf.print(x, [x])

    def receive_output(prediction):
        return tf.print([], [prediction], summarize=4)

    with tfe.protocol.Pond() as prot:

        # treat weights as private
        w = prot.define_private_input('model-provider', provide_weights)

        # load input for prediction
        x = prot.define_private_input('input-provider', provide_input)

        # compute prediction
        y = prot.matmul(x, w)

        # send output
        prediction_op = prot.define_output('input-provider', y, receive_output)

        with tfe.Session() as sess:
            sess.run(tf.global_variables_initializer(), tag='init')

            for _ in range(5):
                sess.run(prediction_op, tag='prediction')
