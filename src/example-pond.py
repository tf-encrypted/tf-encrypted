import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

from tensorflow_encrypted.protocol import Pond, Server
from tensorflow_encrypted.config import session

server0 = Server('/job:localhost/replica:0/task:0/device:CPU:0')
server1 = Server('/job:localhost/replica:0/task:0/device:CPU:1')
crypto_producer = Server('/job:localhost/replica:0/task:0/device:CPU:2')
prot = Pond(server0, server1, crypto_producer)


a = prot.define_constant(np.array([4, 3, 2, 1]).reshape(2,2))
b = prot.define_constant(np.array([4, 3, 2, 1]).reshape(2,2))
c = a * b

d = prot.define_private_variable(np.array([1., 2., 3., 4.]).reshape(2,2))
e = prot.define_private_variable(np.array([1., 2., 3., 4.]).reshape(2,2))
# f = (d * .5 + e * .5)
f = d * e

with session(3) as sess:

    sess.run([d.initializer, e.initializer])

    print f.reveal().eval(sess)

    sess.run(prot.assign(d, f))
    sess.run(prot.assign(e, e))

    print f.reveal().eval(sess)

    g = prot.sigmoid(d)
    print g.reveal().eval(sess)

    # b = prot.define_private_placeholder(shape)

    # c = prot.define_private_variable()
    # d = prot.define_public_variable()
