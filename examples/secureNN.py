import numpy as np
# import tensorflow as tf
# import tensorflow_encrypted as tfe

from tensorflow_encrypted.protocol import SecureNN, Server
from tensorflow_encrypted.config import session

server0 = Server('/job:localhost/replica:0/task:0/device:CPU:0')
server1 = Server('/job:localhost/replica:0/task:0/device:CPU:1')
prot = SecureNN(server0, server1, None)


d = prot.define_private_variable(np.array([1., 2., 3., 4.]).reshape(2, 2))
e = prot.define_private_variable(np.array([1., 2., 3., 4.]).reshape(2, 2))
prot.select_share(d, e)


# apply secureNN functions here

sess = session(3)
sess.run([d.initializer, e.initializer])


sess.close()
