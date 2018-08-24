import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow_encrypted.convert import convert
from tensorflow_encrypted.convert.register import register
import numpy as np
import tensorflow_encrypted as tfe

import os


parser = argparse.ArgumentParser(description='Runs the skin cancer demo with the specified model!')
parser.add_argument('model_path', type=str, help='path to skin cancer model')
args = parser.parse_args()

model_path = args.model_path
with gfile.FastGFile(model_path, 'rb') as f:
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


with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:
    input = PredictionInputProvider(config.get_player('crypto_producer'))

    c = convert.Converter(config, prot, config.get_player('crypto_producer'))
    x = c.convert(graph_def, input, register())

    with config.session() as sess:
        print("initing!!!")
        tfe.run(sess, prot.initializer, tag='init')

        print("running")
        output = x.reveal().eval(sess, tag='prediction')
        print(output)
