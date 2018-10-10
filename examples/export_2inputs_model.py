import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

model_name = "2inputs"
output_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "models",
    "{}.pb".format(model_name),
)
input_data_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "models",
    "{}_input_example.npy".format(model_name),
)

input_size = 16
output_size = 2
input_shape = [1, input_size]
x1 = tf.placeholder(tf.float32, shape=input_shape)
x2 = tf.placeholder(tf.float32, shape=input_shape)

x = x1 + x2
w1 = tf.get_variable("w1", [input_size, output_size],
                     initializer=tf.random_normal_initializer())
b1 = tf.get_variable("b1", [output_size, ],
                     initializer=tf.random_normal_initializer())
x = tf.matmul(x, w1) + b1
y = tf.sigmoid(x)

pred_node_names = ["output"]
pred = [tf.identity(y, name=pred_node_names[0])]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

constant_graph = graph_util.convert_variables_to_constants(
    sess,
    sess.graph.as_graph_def(),
    pred_node_names
)

frozen = graph_util.remove_training_nodes(constant_graph)

graph_io.write_graph(frozen, ".", output_file, as_text=False)
print('saved the frozen graph (ready for inference) at: ', output_file)

data = np.random.standard_normal(input_shape)
np.save(input_data_file, data)
