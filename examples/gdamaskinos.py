import sys
import time
import numpy as np

from functools import partial
from functools import reduce
import tensorflow as tf
import tf_encrypted as tfe
from tensorflow.python.framework import graph_util, graph_io

print(tfe.__file__)
print("Support for int64: ", tfe.config.tensorflow_supports_int64())

tf.set_random_seed(42)
np.random.seed(42)

small = False
if small:
  print("Running CNN with small input")
else:
  print("Running CNN with large input")

config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto-producer',
    'weights-provider',
    'prediction-client'
])

# config = tfe.config.load('config.json')

#config = tfe.RemoteConfig([
#  ('server0', 'server0:4441'),
#  ('server1', 'server1:4442'),
#  ('crypto-producer', 'server2: 4443'),
#  ('weights-provider', 'server3: 4444'),
#  ('prediction-client', 'server2:4442')
#  ], log_device_placement=False)

if len(sys.argv) > 1:
  if isinstance(config, tfe.LocalConfig):
      raise Exception("You can launch a configured server only with a remote configuration")
  player_name = str(sys.argv[1])
  server = config.server(player_name)
  server.start()
  server.join()

tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())

input_shape1 = [1, 3, 32, 32]
input_shape2 = [1, 3, 192, 192]
conv1_fshape = [3, 3, 3, 32]

def provide_input_conv1weights() -> tf.Tensor:
  w = tf.constant(np.random.normal(size=conv1_fshape), dtype=tf.float32)
  return w

def provide_input_denseweights(shape) -> tf.Tensor:
  w = tf.constant(np.random.normal(size=[shape, 2]), dtype=tf.float32)
  return w

def provide_input_prediction() -> tf.Tensor:
  x = tf.random_normal(shape=input_shape1, dtype=tf.float32, seed=42)
  return x

def provide_input_prediction2() -> tf.Tensor:
  x = tf.random_normal(shape=input_shape2, dtype=tf.float32)
  return x

def receive_output(tensor: tf.Tensor) -> tf.Operation:
  return tf.Print(tensor, [tensor, tf.shape(tensor)], message="output:")

def tfe_inference(x):
  conv1 = tfe.layers.Conv2D(x.shape.as_list(), conv1_fshape, 1, "SAME")
  initial_w_conv1 = tfe.define_private_input('weights-provider',
                                             provide_input_conv1weights)
  conv1.initialize(initial_w_conv1)
  x = conv1.forward(x)
  relu1 = tfe.layers.activation.Relu(x.shape.as_list())
  x = relu1.forward(x)
  pool1 = tfe.layers.pooling.MaxPooling2D(x.shape.as_list(), pool_size=2,
                                          strides=2, padding='SAME')

  x = pool1.forward(x)
  shape = reduce(lambda x,y: x*y, x.shape.as_list())
  x = x.reshape([-1, shape])

  part = partial(provide_input_denseweights, shape)
  initial_w_dense = tfe.define_private_input('weights-provider', part)
  x = tfe.matmul(x, initial_w_dense)
  return x

def tf_inference(x, data_format="NHWC"):
  """Tensorflow only supports inference with NHWC; tfe convert needs NCHW"""
  x = tf.nn.conv2d(x, provide_input_conv1weights(), strides=[1,1,1,1],
                   padding="SAME", data_format=data_format)
  x = tf.nn.relu(x)
  x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',
                     data_format=data_format)

  shape = reduce(lambda x,y: x*y, x.shape.as_list())
  x = tf.reshape(x, shape=[-1, shape])
  return tf.matmul(x, provide_input_denseweights(shape))

def export_cnn(x):
  x = tf_inference(x, data_format='NCHW')

  pred_node_names = ["output"]
  tf.identity(x, name=pred_node_names[0])

  with tf.Session() as sess:
    constant_graph = graph_util.convert_variables_to_constants(sess,
                                                               sess.graph.as_graph_def(),
                                                               pred_node_names)
  frozen = graph_util.remove_training_nodes(constant_graph)
  graph_io.write_graph(frozen, ".", "bench_cnn.pb", as_text=False)

if small:
  input_func = provide_input_prediction
else:
  input_func = provide_input_prediction2

with tf.Graph().as_default():
  print("Exporting graph...")
  x = input_func()
  x = tf.placeholder(tf.float32, shape=x.shape)
  export_cnn(x)

with tf.Graph().as_default():
  print("Loading graph...")
  with tf.gfile.GFile("bench_cnn.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default():
  print("TFE")
  c = tfe.convert.Converter(config, tfe.get_protocol(),
                            config.get_player('weights-provider'))
  y = c.convert(graph_def, tfe.convert.register(),
                config.get_player('prediction-client'), input_func)

  #x = tfe.define_private_input('prediction-client', input_func)
  #y = tfe_inference(x)

  prediction_op = tfe.define_output('prediction-client', [y], receive_output)

  with tfe.Session(config=config) as sess:
      print("Initialize tensors")
      sess.run(tf.global_variables_initializer(), tag='init')

      print("Predict")
      for i in range(3):
        t = time.time()
        sess.run(prediction_op, tag='prediction')
        print("Inference time: %g" % (time.time() - t))


with tf.Graph().as_default():
  print("TF")
  x = input_func()
  x = tf.transpose(x, (0,2,3,1)) # NHWC

  prediction_op = tf_inference(x)

  with tf.Session() as sess:
      print("Initialize tensors")
      sess.run(tf.global_variables_initializer())

      print("Predict")
      for i in range(1):
        t = time.time()
        sess.run(prediction_op)
        print("Inference time: %g" % (time.time() - t))