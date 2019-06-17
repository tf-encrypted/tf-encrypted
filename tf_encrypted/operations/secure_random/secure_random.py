"""Secure random sampling."""
import logging
import os

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.errors import NotFoundError
import tf_encrypted as tfe

logger = logging.getLogger('tf_encrypted')

def _try_load_secure_random_module():
  """
  Attempt to load and return secure random module; returns None if failed.
  """
  so_file = '{dn}/operations/secure_random/secure_random_module_tf_{tfv}.so'.format(  # pylint: disable=line-too-long
      dn=os.path.dirname(tfe.__file__),
      tfv=tf.__version__,
  )

  if not os.path.exists(so_file):
    logger.warning(
        ("Falling back to insecure randomness since the required custom op "
         "could not be found for the installed version of TensorFlow. Fix this "
         "by compiling custom ops. Missing file was '%s'"), so_file)
    return None

  try:
    return tf.load_op_library(so_file)

  except NotFoundError as ex:
    logger.warning(
        ("Falling back to insecure randomness since the required custom op "
         "could not be found for the installed version of TensorFlow. Fix this "
         "by compiling custom ops. "
         "Missing file was '%s', error was \"%s\"."), so_file, ex)

  except Exception as ex:  # pylint: disable=broad-except
    logger.error(
        ("Falling back to insecure randomness since an error occurred loading "
         "the required custom op: \"%s\"."), ex)

  return None

secure_random_module = _try_load_secure_random_module()

def supports_secure_randomness():
  return secure_random_module is not None


def supports_seeded_randomness():
  return secure_random_module is not None


def seeded_random_uniform(shape,
                          minval=0,
                          maxval=None,
                          dtype=tf.int32,
                          seed=None,
                          name=None):
  """
  Returns cryptographically strong random numbers with a seed

  .. code-block:: python

      x = secure_random([2, 2], minval=-1000, maxval=1000)

  :param list shape: Shape of the random tensor.
  :param int minval: Minimum value to return, inclusive.
  :param int maxval: Maximum value to return, exclusive.
  :param dtype: Data type of the return random values. Either int32 or int64.
  :param tf.Tensor seed: The seed to be used when generating the random numbers.
  :param str name:

  :rtype: tf.Tensor
  """
  dtype = dtypes.as_dtype(dtype)
  if dtype not in (dtypes.int32, dtypes.int64):
    raise ValueError("Invalid dtype %r" % dtype)

  if maxval is None:
    raise ValueError("Must specify maxval for integer dtype %r" % dtype)

  if seed is None:
    raise ValueError("Seed must be passed")

  minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
  maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")

  return secure_random_module.secure_seeded_random_uniform(
      shape,
      seed,
      minval,
      maxval,
      name=name,
  )


def random_uniform(shape, minval=0, maxval=None, dtype=tf.int32, name=None):
  """
  Returns cryptographically strong random numbers.

  .. code-block:: python

      x = secure_random([2, 2], minval=-1000, maxval=1000)

  :param list shape: Shape of the random tensor.
  :param int minval: Minimum value to return, inclusive.
  :param int maxval: Maximum value to return, exclusive.
  :param dtype: Data type of the return random values. Either int32 or int64.
  :param tf.Tensor seed: A seed for generating the random numbers.
  :param str name:

  :rtype: tf.Tensor
  """

  dtype = dtypes.as_dtype(dtype)
  if dtype not in (dtypes.int32, dtypes.int64):
    raise ValueError("Invalid dtype %r" % dtype)

  if maxval is None:
    raise ValueError("Must specify maxval for integer dtype %r" % dtype)

  minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
  maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")

  return secure_random_module.secure_random_uniform(
      shape,
      minval,
      maxval,
      name=name,
  )


def secure_seed():
  return secure_random_module.secure_seed()
