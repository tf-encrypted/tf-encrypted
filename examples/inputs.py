import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

config = tfe.LocalConfig(5)
# config = tfe.RemoteConfig(
#     player_hosts=[
#         'localhost:4440',
#         'localhost:4441',
#         'localhost:4442'
#     ]
# )

class WeightsInputProvider(tfe.io.InputProvider):

    def provide_input(self) -> tf.Tensor:
        raw_weights = np.array([1, 2, 3, 4]).reshape((2,2))
        return tf.constant(raw_weights)

class PredictionInputProvider(tfe.io.InputProvider):

    def provide_input(self) -> tf.Tensor:
        return tf.constant([1, 2, 3, 4], shape=(2,2), dtype=tf.float32)

class PredictionOutputReceiver(tfe.io.OutputReceiver):

    def receive_output(self, tensor: tf.Tensor) -> tf.Operation:
        return tf.Print(tensor, [tensor])

weights_input = WeightsInputProvider(config.players[3])
prediction_input = PredictionInputProvider(config.players[3])
prediction_output = PredictionOutputReceiver(config.players[4])

with tfe.protocol.Pond(*config.players[:3]) as prot:

    # # treat weights as private
    # initial_w = prot.define_private_input(weights_input)
    # w = prot.define_private_variable(initial_w)

    # treat weights as private, but initial value as public
    # initial_w = prot.define_public_input(weights_input)
    # w = prot.define_private_variable(initial_w)

    # treat weights as public
    initial_w = prot.define_public_input(weights_input)
    w = prot.define_public_variable(initial_w)

    # load input for prediction
    x = prot.define_private_input(prediction_input)

    # compute prediction
    y = x.dot(w)

    # send output
    prediction_op = prot.define_output(y, prediction_output)

    with config.session() as sess:
        tfe.run(sess, tf.global_variables_initializer(), tag='init')

        for _ in range(5):
            tfe.run(sess, prediction_op, tag='prediction')
