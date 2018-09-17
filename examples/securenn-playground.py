import numpy as np
import tensorflow_encrypted as tfe

config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto_producer'
])

with tfe.protocol.SecureNN(*config.get_players('server0, server1, crypto_producer')) as prot:

    a = prot.define_constant(np.array([0, 0, 1, 1]), apply_scaling=False)
    b = prot.define_constant(np.array([0, 1, 0, 1]), apply_scaling=False)
    c = prot.bitwise_or(a, b)

    x = prot.define_constant(np.array([0., 1., 2., 3.]))
    y = prot.define_constant(np.array([0., 1., 2., 3.]))
    z = (x * c) * y

    with config.session() as sess:

        print(z.eval(sess, tag='res'))
