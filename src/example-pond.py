import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

from tensorflow_encrypted.protocol import Pond, Server
from tensorflow_encrypted.config import session

server0 = Server('/job:localhost/replica:0/task:0/device:CPU:0')
server1 = Server('/job:localhost/replica:0/task:0/device:CPU:1')
prot = Pond(server0, server1, None)

shape = (2,2)

a = prot.define_public_placeholder(shape)
b = a * 2

c = mask(a) + b


# b = prot.define_private_placeholder(shape)

# c = prot.define_private_variable()
# d = prot.define_public_variable()
