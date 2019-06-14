import tensorflow as tf

import tf_encrypted as tfe
from collections import OrderedDict

from tf_encrypted.operations import ntl


remote_config = tfe.RemoteConfig(
    OrderedDict([
        ('master', "localhost:4045"),
        ('server0', "localhost:4041"),
        ('server1', "localhost:4042"),
    ])
)

tfe.set_config(remote_config)

tfe.set_protocol(tfe.protocol.Pond())

prot = tfe.get_protocol()

server = remote_config.server("master")

with tfe.Session() as sess:

    with tf.device(prot.server_0.device_name):
      m1 = ntl.create_ntl_matrix([[5, 5], [5, 5]], 5000000)
      m2 = ntl.create_ntl_matrix([[5, 5], [5, 5]], 5000000)

    res = ntl.matmul_ntl(m1, m2, 5000000)
    n = ntl.ntl_to_native(res, tf.int32)
    print(sess.run(n))
