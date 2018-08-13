import numpy as np
import tensorflow_encrypted as tfe

from tensorflow_encrypted.protocol import Pond, Player
from tensorflow_encrypted.config import local_session

server0 = Player('/job:localhost/replica:0/task:0/device:CPU:0')
server1 = Player('/job:localhost/replica:0/task:0/device:CPU:1')
crypto_producer = Player('/job:localhost/replica:0/task:0/device:CPU:2')
prot = Pond(server0, server1, crypto_producer)

# parameters
np_w = np.array([.1, .2, .3, .4]).reshape(2,2)
np_b = np.array([.1, .2, .3, .4]).reshape(2,2)
w = prot.define_private_variable(np_w)
b = prot.define_private_variable(np_b)

# input
x = prot.define_private_placeholder((2,2))

# prediction
y = prot.sigmoid(w.dot(x) + b).reveal()

def sigmoid (x):
    return 1/(1 + np.exp(-x))

with local_session(3) as sess:

    tfe.run(sess, prot.initializer, tag='init')

    np_x = np.array([.1, -.1, .2, -.2]).reshape(2,2)
    feed_dict = x.feed_from_native(np_x)
    prediction = y.eval(sess, feed_dict=feed_dict, tag='reveal')

    print('tf pred: ', prediction)
    print('numpy pred: ', sigmoid(np_w.dot(np_x) + np_b))
