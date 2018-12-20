import sys
import time
import numpy as np

from functools import partial
from functools import reduce
import tensorflow as tf
import tf_encrypted as tfe
from tensorflow.python.framework import graph_util, graph_io


config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto-producer',
    'weights-provider',
    'prediction-client'
])

# config = tfe.config.load('config.json')


tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())

a = tf.random_uniform(shape=[500, 500, 50], minval=0, maxval=1000000000, dtype=tf.int64)
x = tfe.tensor.int64factory.tensor(a)
y = x.bits()

with tfe.Session(config=config) as sess:
    for i in range(3):
        sess.run(y, tag='bits')
