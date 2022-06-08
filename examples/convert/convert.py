import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

import tf_encrypted as tfe
from tf_encrypted.convert import convert
from tf_encrypted.convert.register import registry


def export_cnn() -> None:
    input = tf.placeholder(tf.float32, shape=(1, 1, 28, 28))
    filter = tf.constant(np.ones((3, 3, 1, 1)), dtype=tf.float32)
    x = tf.nn.conv2d(input, filter, (1, 1, 1, 1), "SAME", data_format="NCHW")
    a = tf.nn.sigmoid(x)
    b = tf.nn.relu(x)
    c = tf.keras.layers.Conv2D(1, 3, padding="same", data_format="channels_first")(a)
    d = tf.keras.layers.Conv2D(1, 3, padding="same", data_format="channels_first")(b)
    e = tf.keras.layers.Conv2D(1, 3, padding="same", data_format="channels_first")(x)
    with tf.name_scope("scope1"):
        f = tf.keras.layers.Conv2D(1, 3, padding="same", data_format="channels_first")(
            a
        )
    with tf.name_scope("scope2"):
        g = tf.keras.layers.Conv2D(1, 3, padding="same", data_format="channels_first")(
            b
        )
    x = c + d + e + f + g
    x = tf.keras.layers.BatchNormalization(axis=1)(x, training=False)

    pred_node_names = ["output"]
    tf.identity(x, name=pred_node_names[0])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        constant_graph = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), pred_node_names
        )

    frozen = graph_util.remove_training_nodes(constant_graph)

    output = "cnn.pb"
    graph_io.write_graph(frozen, ".", output, as_text=False)


export_cnn()

tf.reset_default_graph()

model_filename = "cnn.pb"
with gfile.GFile(model_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

config = tfe.LocalConfig(
    ["server0", "server1", "crypto-producer", "prediction-client", "weights-provider"]
)
tfe.set_config(config)


def provide_input() -> tf.Tensor:
    return tf.constant(np.random.normal(size=(1, 1, 28, 28)), tf.float32)


def receive_output(tensor: tf.Tensor) -> tf.Tensor:
    tf.print(tensor, [tensor])
    return tensor


with tfe.protocol.Pond(
    *config.get_players("server0, server1, crypto-producer")
) as prot:

    c = convert.Converter(
        registry(),
        config=config,
        protocol=prot,
        model_provider=config.get_player("weights-provider"),
    )
    x = c.convert(graph_def, config.get_player("prediction-client"), provide_input)

    prediction_op = prot.define_output(
        config.get_player("prediction-client"), x, receive_output
    )

    with tfe.Session(config=config) as sess:
        sess.run(tfe.global_variables_initializer(), tag="init")

        sess.run(prediction_op, tag="prediction")

# os.remove(model_filename)
