import numpy as np
import tensorflow_encrypted as tfe

config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto_producer'
])

with tfe.protocol.SecureNN(*config.get_players('server0, server1, crypto_producer')) as prot:

    a = prot.define_constant(np.array([0, 0, 1, 1]).reshape(2,2), encode=False)
    b = prot.define_constant(np.array([0, 1, 0, 1]).reshape(2,2), encode=False)
    c = prot.bitwise_and(a, b)
    
    with config.session() as sess:

        print(c.eval(sess, tag='res'))
