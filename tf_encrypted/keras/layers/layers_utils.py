"""TF Encrypted Keras layers utils"""
import inspect

import tensorflow as tf


def not_implemented_arg_err(arg, arg_name, tf_layer_name):
  """Raise error when layer argument is not yet supported in TFE
    Arguments:
      arg: TFE layer argument
      arg_name: TFE layer argument name (string)
      tf_layer_name: Tensorflow keras layer name (string)
  """
  arg_not_impl_msg = ("`{}` argument is not implemented for layer {}. "
                      "Please use {} as a default value")

  tf_layer_cls = getattr(tf.keras.layers, tf_layer_name)
  layer_sign = inspect.signature(tf_layer_cls.__init__).parameters
  tf_default_arg = layer_sign[arg_name].default

  if arg != tf_default_arg:
    raise NotImplementedError(
        arg_not_impl_msg.format(arg_name, tf_layer_name, tf_default_arg)
    )
  