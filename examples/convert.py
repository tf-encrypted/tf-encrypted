import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.keras as K

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow_encrypted.convert import convert
from tensorflow_encrypted.convert.register import register
import numpy as np
import tensorflow_encrypted as tfe


def export_cnn():
    input = tf.placeholder(tf.float32, shape=(1, 1, 28, 28))
    filter = tf.constant(np.ones((5, 5, 1, 16)), dtype=tf.float32)
    x = tf.nn.conv2d(input, filter, (1, 1, 1, 1), "SAME", data_format='NCHW')
    x = tf.nn.sigmoid(x)
    x = tf.nn.relu(x)

    sess = K.backend.get_session()

    pred_node_names = ["output"]
    tf.identity(x, name=pred_node_names[0])

    constant_graph = graph_util.convert_variables_to_constants(sess,
                                                               sess.graph.as_graph_def(),
                                                               pred_node_names)

    frozen = graph_util.remove_training_nodes(constant_graph)

    output = "cnn.pb"
    graph_io.write_graph(frozen, ".", output, as_text=False)
    print('saved the frozen graph (ready for inference) at: ', output)


class PredictionInputProvider(tfe.io.InputProvider):
    def provide_input(self) -> tf.Tensor:
        return tf.constant(np.random.normal(size=(1, 1, 28, 28)), tf.float32)


class PredictionOutputReceiver(tfe.io.OutputReceiver):
    def receive_output(self, tensor: tf.Tensor) -> tf.Operation:
        return tf.Print(tensor, [tensor])


export_cnn()

tf.reset_default_graph()

model_filename = 'cnn.pb'
with gfile.FastGFile(model_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto_producer',
    'prediction_client',
    'weights_provider'
])

with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:
    input = PredictionInputProvider(config.get_player('prediction_client'))
    output = PredictionOutputReceiver(config.get_player('prediction_client'))

    c = convert.Converter(config, prot, config.get_player('weights_provider'))
    x = c.convert(graph_def, input, register())

    prediction_op = prot.define_output(x, output)

    with config.session() as sess:
        tfe.run(sess, prot.initializer, tag='init')

        tfe.run(sess, prediction_op, tag='prediction')
