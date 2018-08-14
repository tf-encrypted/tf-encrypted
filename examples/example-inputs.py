import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

config = tfe.LocalConfig(3)
# config = tfe.RemoteConfig(
#     player_hosts=[
#         'localhost:4440',
#         'localhost:4441',
#         'localhost:4442'
#     ]
# )

with tfe.protocol.Pond(*config.players) as prot:

    w = prot.define_private_variable(initial_value=np.ones((10,1)))
    x = prot.define_private_placeholder(shape=(1,10))
    y = x.dot(w)

    with config.session() as sess:

        # initialize variables
        tfe.run(sess, prot.initializer, tag='init')

        # prepare feed
        feed_dict = x.feed_from_native(np.arange(10).reshape(1,10))

        # run prediction
        res = y.reveal().eval(sess, feed_dict=feed_dict, tag='predict')
        print(res)
