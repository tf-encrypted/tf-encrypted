import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

output_file = dir_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "models",
    "matmul.pb",
)

input_shape = [1, 16]
x = tf.placeholder(tf.float32, shape=input_shape)

w1 = tf.get_variable("w1", [16, 16])
b1 = tf.get_variable("b1", [16, ])
x = tf.matmul(x, w1) + b1

w2 = tf.get_variable("w2", [16, 2])
y = tf.matmul(x, w2)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
pred_node_names = ["output"]
pred = [tf.identity(y, name=pred_node_names[0])]

constant_graph = graph_util.convert_variables_to_constants(
    sess,
    sess.graph.as_graph_def(),
    pred_node_names
)

frozen = graph_util.remove_training_nodes(constant_graph)

graph_io.write_graph(frozen, ".", output_file, as_text=False)
print('saved the frozen graph (ready for inference) at: ', output_file)
