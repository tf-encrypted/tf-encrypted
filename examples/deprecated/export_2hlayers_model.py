import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util

model_name = "2lfc"
output_file = dir_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "models",
    "{}.pb".format(model_name),
)
input_data_file = dir_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "models",
    "{}_input_example.npy".format(model_name),
)

input_size = 16
hidden_size = 512
input_shape = [1, input_size]
x = tf.placeholder(tf.float32, shape=input_shape)

w1 = tf.get_variable(
    "w1", [input_size, hidden_size], initializer=tf.random_normal_initializer()
)
b1 = tf.get_variable("b1", [hidden_size], initializer=tf.random_normal_initializer())
x = tf.matmul(x, w1) + b1
x = tf.sigmoid(x)

w2 = tf.get_variable(
    "w2", [hidden_size, hidden_size], initializer=tf.random_normal_initializer()
)
b2 = tf.get_variable("b2", [hidden_size])
x = tf.matmul(x, w2) + b2
x = tf.sigmoid(x)

w3 = tf.get_variable("w3", [hidden_size, 2], initializer=tf.random_normal_initializer())
y = tf.matmul(x, w3)

pred_node_names = ["output"]
pred = [tf.identity(y, name=pred_node_names[0])]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

constant_graph = graph_util.convert_variables_to_constants(
    sess, sess.graph.as_graph_def(), pred_node_names
)

frozen = graph_util.remove_training_nodes(constant_graph)

graph_io.write_graph(frozen, ".", output_file, as_text=False)
print("saved the frozen graph (ready for inference) at: ", output_file)

data = np.random.standard_normal(input_shape)
np.save(input_data_file, data)
