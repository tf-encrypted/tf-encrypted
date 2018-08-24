import tensorflow as tf
from tensorflow.python.platform import gfile

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow_encrypted.convert import convert
from tensorflow_encrypted.convert.register import register
import numpy as np
import tensorflow_encrypted as tfe

import os


model_filename = 'skin_cancer_tensorflow_model.pb'
with gfile.FastGFile(model_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto_producer'
])


class PredictionInputProvider(tfe.io.InputProvider):
    def provide_input(self) -> tf.Tensor:
        return tf.constant(np.random.normal(size=(1, 200, 200, 3)), tf.float32)


class PredictionOutputReceiver(tfe.io.OutputReceiver):
    def receive_output(self, tensor: tf.Tensor) -> tf.Operation:
        return tf.Print(tensor, [tensor])


with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:
    input = PredictionInputProvider(config.get_player('crypto_producer'))
    output = PredictionOutputReceiver(config.get_player('crypto_producer'))

    c = convert.Converter(config, prot, config.get_player('crypto_producer'))
    x = c.convert(graph_def, input, register())

    prediction_op = prot.define_output(x, output)

    with config.session() as sess:
        print("initing!!!")
        tfe.run(sess, prot.initializer, tag='init')

        print("running")
        print(tfe.run(sess, prediction_op, tag='prediction'))
