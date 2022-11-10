"""TF Encrypted utilities."""
import inspect

import tensorflow as tf


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
            "Don't know how to process {} of type {}".format(node, type(node))
        )

    return reachable


def unwrap_fetches(fetches):
    """
    Unwraps TF Encrypted fetches into TensorFlow-compatible fetches.
    """

    if isinstance(fetches, (list, tuple)):
        return [unwrap_fetches(fetch) for fetch in fetches]
    if isinstance(fetches, (tf.Tensor, tf.Operation, tf.Variable)):
        return fetches
    try:
        native = getattr(fetches, "to_native")
        return native()
    except AttributeError:
        return fetches


def print_banner(title):
    title_length = len(title)
    banner_length = title_length + 2 * 10
    banner_top = "+" + ("-" * (banner_length - 2)) + "+"
    banner_middle = "|" + " " * 9 + title + " " * 9 + "|"

    print()
    print(banner_top)
    print(banner_middle)
    print(banner_top)
