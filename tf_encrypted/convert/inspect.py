"""Methods for inspecting TF and Keras graphs for """
import os
import tempfile

import tensorflow as tf


def inspect_subgraph(subgraph, input_shape, sess=None):
  path = _gen_graph_def(subgraph, input_shape, sess)
  graph = _read_graph(path)
  print_graph(graph)


def _gen_graph_def(subgraph, input_shape, sess):
  temp_dir = tempfile.gettempdir()
  filename = "inspect_{}.pb".format(subgraph.__class__.__name__)
  filepath = os.path.join(temp_dir, filename)

  if isinstance(input_shape, tuple):
    input_shape = [1] + list(input_shape)
  x = tf.zeros(input_shape)
  y = subgraph(x)
  return export(y, filepath, sess=sess)


def _read_graph(path):
  with tf.io.gfile.GFile(path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  return graph_def


def print_from_graphdef(graphdef):
  for node in graphdef.node:
    print("| Name:", node.name, "| Op:", node.op, "|")


def export(x: tf.Tensor, filename, sess=None):
  """Export a GraphDef for the graph connected to x.

  Args:
    x: The tensor whose parent Graph we want to export.
    filename: The filename to use for the exported GraphDef.
    sess: An optional Session object; useful if already operating within a
        Session context.

  Returns:
    A filepath pointing to the exported GraphDef.
  """
  should_close = False
  if sess is None:
    should_close = True
    sess = tf.Session()

  pred_node_names = ["output"]
  tf.identity(x, name=pred_node_names[0])
  graph = tf.graph_util.convert_variables_to_constants(
      sess,
      sess.graph.as_graph_def(),
      pred_node_names
    )

  graph = tf.graph_util.remove_training_nodes(graph)

  path = tf.io.write_graph(graph, ".", filename, as_text=False)

  if should_close:
    sess.close()

  return path
