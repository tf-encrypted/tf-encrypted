import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

from tensorflow_encrypted.protocol import Pond, Server
from tensorflow_encrypted.config import session
from tensorflow_encrypted.layer import Conv2D, set_protocol

server0 = Server('/job:localhost/replica:0/task:0/device:CPU:0')
server1 = Server('/job:localhost/replica:0/task:0/device:CPU:1')
crypto_producer = Server('/job:localhost/replica:0/task:0/device:CPU:2')
prot = Pond(server0, server1, crypto_producer)

# TODO[koen]: temporary ugly hack
set_protocol(prot)


a = prot.define_constant(np.array([4, 3, 2, 1]).reshape(2, 2))
b = prot.define_constant(np.array([4, 3, 2, 1]).reshape(2, 2))
c = a * b

d = prot.define_private_variable(np.array([1., 2., 3., 4.]).reshape(2, 2))
e = prot.define_private_variable(np.array([1., 2., 3., 4.]).reshape(2, 2))
# f = (d * .5 + e * .5)
f = d * e


# convolutions
conv_input_shape = (32, 1, 28, 28)  #NCHW
conv_input = prot.define_private_variable(np.random.normal(size=conv_input_shape))
conv_layer = Conv2D((4, 4, 1, 20), strides=2)
conv_layer.initialize(conv_input_shape)
conv_out = conv_layer.forward(conv_input)


with session(3) as sess:

    sess.run([d.initializer, e.initializer])
    print "multiplication : "
    print f.reveal().eval(sess)

    print "assignment : "
    sess.run(prot.assign(d, f))
    sess.run(prot.assign(e, e))

    print "multiplication after assignment : "
    print f.reveal().eval(sess)

    print "sigmoid: "
    g = prot.sigmoid(d)
    print g.reveal().eval(sess)

    print "convolution forward: "
    sess.run([conv_layer.weights.initializer, conv_layer.bias.initializer, conv_input.initializer])
    print conv_out.reveal().eval(sess)



    # b = prot.define_private_placeholder(shape)
    # c = prot.define_private_variable()
    # d = prot.define_public_variable()
