"""TF Encrypted utilities."""
import tensorflow as tf


def wrap_in_variables(*tensors):
  """Wrap a list of tensors in Variables"""
  variables = [
      tensor.factory.variable(
          tf.zeros(shape=tensor.shape, dtype=tensor.factory.native_type)
      )
      for tensor in tensors
  ]
  group_updater = tf.group(
      *[var.assign_from_same(tensor) for var, tensor in zip(variables, tensors)]
  )
  return group_updater, variables


def flatten(xs):
  """
  Flatten any recursive list or tuple into a single list.

  For instance:
  - `flatten(x) => [x]`
  - `flatten([x]) => [x]`
  - `flatten([x, [y], [[z]]]) => `[x, y, z]`
  """
  if isinstance(xs, (list, tuple)):
    return [y for ys in [flatten(x) for x in xs] for y in ys]
  return [xs]


def reachable_nodes(*nodes):
  """
  Find all nodes reachable from `nodes` in the implicit tf.Graph
  to which they belong.

  Both tensors and their underlying operation is returned.
  """

  nodes = flatten(nodes)
  reachable = set(nodes)
  queue = list(nodes)

  while queue:
    node = queue.pop(0)

    if isinstance(node, tf.Tensor):
      subnode = node.op
      if subnode not in reachable:
        reachable.add(subnode)
        queue.append(subnode)
      continue

    if isinstance(node, tf.Operation):
      for subnode in list(node.inputs) + list(node.control_inputs):
        if subnode not in reachable:
          reachable.add(subnode)
          queue.append(subnode)
      continue

    raise TypeError(
        "Don't know how to process {} of type {}".format(node, type(node)))

  return reachable


def unwrap_fetches(fetches):
  """
  Unwraps TF Encrypted fetches into TensorFlow-compatible fetches.
  """

  if isinstance(fetches, (list, tuple)):
    return [unwrap_fetches(fetch) for fetch in fetches]
  if isinstance(fetches, (tf.Tensor, tf.Operation)):
    return fetches
  return fetches.to_native()
