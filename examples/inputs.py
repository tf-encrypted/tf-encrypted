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

class PredictionInputProvider(object):

    def __init__(self, player: tfe.protocol.Player):
        self.player = player

    def provide_input(self) -> tf.Tensor:
        return tf.constant([1, 2, 3, 4], shape=(2,2), dtype=tf.float32)

class PredictionOutputReceiver(object):

    def __init__(self, player: tfe.protocol.Player):
        self.player = player
    
    def receive_output(self, tensor):
        return tf.Print(tensor, tensor)

prediction_input = PredictionInputProvider(config.players[3])
# prediction_output = PredictionOutputReceiver(config.players[4])

with tfe.protocol.Pond(*config.players[:3]) as prot:

    w = prot.define_private_variable(np.array([1, 2, 3, 4]).reshape((2,2)))
    x = prot.define_private_input(prediction_input)
    y = x.dot(w)

    # prediction_op = prot.define_private_output(y, prediction_output)
    
    with config.session() as sess:
        tfe.run(sess, tf.global_variables_initializer(), tag='init')

        # tfe.run(sess, prediction_op, tag='prediction')
        
        res = y.reveal().eval(sess, tag='prediction')
        print(res)
