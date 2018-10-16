import numpy as np
import tf_encrypted as tfe

config = tfe.get_config()
with tfe.protocol.SecureNN(*config.get_players('server0, server1, crypto_producer')) as prot:

    a = prot.define_constant(np.array([0, 0, 1, 1]), apply_scaling=False)
    b = prot.define_constant(np.array([0, 1, 0, 1]), apply_scaling=False)
    c = prot.bitwise_or(a, b)

    x = prot.define_constant(np.array([0., 1., 2., 3.]))
    y = prot.define_constant(np.array([0., 1., 2., 3.]))
    z = (x * c) * y

    with tfe.Session() as sess:
        print(sess.run(z, tag='res'))
