import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

from tensorflow_encrypted.protocol import Pond, Server

# local
server0 = Server('/job:localhost/replica:0/task:0/device:CPU:0')
server1 = Server('/job:localhost/replica:0/task:0/device:CPU:1')
server2 = Server('/job:localhost/replica:0/task:0/device:CPU:2')

# remote
# master = '10.0.0.1:4440'
# server0 = Server('/job:spdz/replica:0/task:0/cpu:0')
# server1 = Server('/job:spdz/replica:0/task:1/cpu:0')
# server2 = Server('/job:spdz/replica:0/task:2/cpu:0')

prot = Pond(server0, server1, server2)

w = prot.define_private_variable(np.zeros((100,100)))
# x = prot.define_private_variable(np.zeros((1,100)))

y = w
for _ in range(40):
    y = y.dot(y)

with tfe.local_session(3) as sess:
# with tfe.remote_session(master) as sess:
    tfe.run(sess, prot.initializer, tag='init')
    print(y.reveal().eval(sess, tag='reveal'))
