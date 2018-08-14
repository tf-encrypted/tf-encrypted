import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.keras as K

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow_encrypted.convert import convert
import numpy as np

import os


def export_vgg16():
    vgg16 = tf.keras.applications.VGG16()

    sess = K.backend.get_session()

    pred = [None]
    pred_node_names = [""]
    pred_node_names[0] = "output"
    pred[0] = tf.identity(vgg16.outputs[0], name=pred_node_names[0])

    constant_graph = graph_util.convert_variables_to_constants(sess,
                                                               sess.graph.as_graph_def(),
                                                               pred_node_names)

    frozen = graph_util.remove_training_nodes(constant_graph)

    output = "vgg16.pb"
    graph_io.write_graph(frozen, ".", output, as_text=False)
    print('saved the frozen graph (ready for inference) at: ', output)


export_vgg16()

tf.reset_default_graph()

model_filename = 'vgg16.pb'
with gfile.FastGFile(model_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

output_vars = convert(graph_def)

input = np.ones((1, 224, 224, 3))

with tf.Session() as sess:
    pred = sess.run(output_vars[graph_def.node[-1].name],
                    feed_dict={output_vars[graph_def.node[0].name]: input})

    print(pred)

os.remove('vgg16.pb')
