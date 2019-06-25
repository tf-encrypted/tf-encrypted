"""TF Encrypted Keras layers utils"""
import inspect

import tensorflow as tf


class LayerArgNotImplementedError(NotImplementedError):
  """Raise error when layer argument is not yet supported in TFE

    Args:
      arg: TFE layer argument
      arg_name: TFE layer argument name (string)
      tf_layer_name: Tensorflow keras layer name (string)
  """
  def __init__(self, arg_name, tf_layer_name, tf_default_arg):
    super(LayerArgNotImplementedError, self).__init__()
    self.arg_name = arg_name
    self.tf_layer_name = tf_layer_name
    self.tf_default_arg = tf_default_arg

  def __str__(self):
    arg_not_impl_msg = ("`{}` argument is not implemented for layer {}. "
                        "Please use the default value of {}.")
    return arg_not_impl_msg.format(self.arg_name,
                                   self.tf_layer_name,
                                   self.tf_default_arg)


def default_args_check(arg, arg_name, tf_layer_name):
  """Check if the layer is using the dfault argument

    Args:
      arg: TFE layer argument
      arg_name: TFE layer argument name (string)
      tf_layer_name: Tensorflow keras layer name (string)

    Raises:
      NotImplementedError: if this argument is not implemented for this `layer`.
  """
  tf_layer_cls = getattr(tf.keras.layers, tf_layer_name)
  layer_sign = inspect.signature(tf_layer_cls.__init__).parameters
  tf_default_arg = layer_sign[arg_name].default
  if arg != tf_default_arg:
    raise LayerArgNotImplementedError(arg_name, tf_layer_name, tf_default_arg)
