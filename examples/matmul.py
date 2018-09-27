import numpy as np
import tensorflow_encrypted as tfe

# use local config (for development/debugging)
config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto_producer'
])

# use remote config
# config = tfe.RemoteConfig([
#     'localhost:4440',
#     'localhost:4441',
#     'localhost:4442'
# ])

# use remote config from cluster file
# config = tfe.RemoteConfig.from_file('cluster.json')

with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:

    w = prot.define_private_variable(np.zeros((10, 10)))
    # x = prot.define_private_variable(np.zeros((1,100)))

    y = w
    for _ in range(5):
        y = y.matmul(y)

    with tfe.Session() as sess:
        sess.run(prot.initializer, tag='init')
        print(y.reveal().eval(sess, tag='reveal'))
