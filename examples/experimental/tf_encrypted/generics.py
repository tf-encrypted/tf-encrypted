"""Core tfe ops.

The tf_encrypted namespace should contain all of these functions.

They can be generated automatically and generically from the TF module,
however I've included a few specific functions here for clarity.
"""
from tf_encrypted.protocol import dispatch

####################
##  Illustrative  ##
####################

def add(x, y):
  context = tf.get_replica_context()

  protocol_func = dispatch(context, 'add', x, y)

  if protocol_func is not None:
    return protocol_func(context, x, y)  # pylint: disable=not-callable

  raise TypeError("Don't know how to add {} and {}".format(type(x), type(y)))


def mul(x, y):
  context = tf.get_replica_context()

  protocol_func = dispatch(context, 'mul', x, y)

  if protocol_func is not None:
    return protocol_func(context, x, y)  # pylint: disable=not-callable

  raise TypeError("Don't know how to mul {} and {}".format(type(x), type(y)))

####################
##    Generics    ##
####################

def build_tfe_op(op):
  """Enable dispatching from higher-level tfe ops.

  This function creates an op for the main tf_encrypted namespace.  That op
  becomes responsible for dispatching calls to underlying kernel functions,
  based on the current protocol (i.e. replica_context), and dispatch_id
  (i.e. privacy) of the combined args and kwargs.

  Example usage:
    In tf_encrypted._init__.py: `tfe.add = build_tfe_op('add')`

    In script:
    ```
    x = PublicTensor(...)
    y = PrivateTensor(...)
    with tfe.Pond():
      z_pond = tfe.add(x, y)  # dispatched to _pond_add_public_private
    with tfe.Seal():
      z_seal = tfe.add(x, y)  # dispatched to _seal_add_public_private
     ```

  Args:
    op (str): The name of the op.

  Returns:
    The higher-level TFE op function.
  """

  def generic_tfe_command(*args, **kwargs):
    context = tf.get_replica_context()

    protocol_func = dispatch(context, op_str, *args, **kwargs)

    if protocol_func is not None:
      return protocol_func(*args, **kwargs)

    raise TypeError(("Don't know how to {}: "
                     "{}").format(base_name, [type(arg) for arg in args]))

  return generic_tfe_command
