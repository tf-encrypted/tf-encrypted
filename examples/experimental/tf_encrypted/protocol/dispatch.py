import tensorflow as tf
import tf_encrypted.protocol as protocols_module


def dispatch(context, base_name, *args, **kwargs):
  protocol_name = context.dispatch_id
  suffix = "_".join([arg.dispatch_id
                     for arg in args if hasattr(arg, "dispatch_id")])

  func_name = "_{}_{}_{}".format(protocol_name, base_name, suffix)

  protocol_module = getattr(protocols_module, protocol_name)
  container = getattr(protocol_module, 'kernels')
  func = getattr(container, func_name, None)

  return func
