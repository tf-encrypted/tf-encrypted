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

a = np.ones((10, 10))

x = tfe.define_private_variable(a)

b = a
y = x
for _ in range(2):
    b = np.dot(b, b)
    y = y.matmul(y)

with tfe.Session() as sess:
    sess.run(tfe.global_variables_initializer(), tag='init')
    actual = sess.run(y.reveal(), tag='reveal')

    expected = b
    np.testing.assert_allclose(actual, expected, atol=.1)
