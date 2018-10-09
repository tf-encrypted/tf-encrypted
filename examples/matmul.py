import numpy as np
import tensorflow_encrypted as tfe

# use local config (for development/debugging)
# A default configuration exists if you don't want to
# config = tfe.LocalConfig([
#     'server0',
#     'server1',
#     'crypto_producer'
# ])

# use remote config
# config = tfe.RemoteConfig([
#     'localhost:4440',
#     'localhost:4441',
#     'localhost:4442'
# ])

# use remote config from cluster file
# config = tfe.RemoteConfig.from_file('cluster.json')

# Setting your custom configuration
# tfe.set_config(config)

# TFE sets a default cryptographic protocol, called Pond
# It's a 3 party computation protocol
# But you can use your another one!
# tfe.set_protocol(tfe.protocol.SecureNN())

w = tfe.define_private_variable(np.ones((10, 10)))

y = w
for _ in range(2):
    y = y.matmul(y)

with tfe.Session() as sess:
    sess.run(tfe.global_variables_initializer(), tag='init')
    print(sess.run(y.reveal(), tag='reveal'))
