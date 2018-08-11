import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

from tensorflow_encrypted.protocol import Pond, Server
from tensorflow_encrypted.config import local_session

server0 = Server('/job:localhost/replica:0/task:0/device:CPU:0')
server1 = Server('/job:localhost/replica:0/task:0/device:CPU:1')
crypto_producer = Server('/job:localhost/replica:0/task:0/device:CPU:2')
prot = Pond(server0, server1, crypto_producer)

input_provider =

# parameters
w = prot.define_private_variable(np.array([1., 2., 3., 4.]).reshape(2,2))
b = prot.define_private_variable(np.array([1., 2., 3., 4.]).reshape(2,2))

# input
x = prot.define_private_placeholder((2,2))

# prediction
y = prot.sigmoid(w.dot(x) + b).reveal()

with local_session(3) as sess:

    tfe.run(sess, prot.initializer, tag='init')

    feed_dict = x.feed_from_native(np.array([1., 1., 1., 1.]).reshape(2,2))
    prediction = y.eval(sess, feed_dict=feed_dict, tag='reveal')

    print prediction