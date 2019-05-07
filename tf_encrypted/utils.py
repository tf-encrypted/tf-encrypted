import tensorflow as tf


def wrap_in_variables(*tensors):
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
