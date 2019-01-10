import logging
import os

import tensorflow as tf
import tf_encrypted as tfe

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.errors import NotFoundError


dirname = os.path.dirname(tfe.__file__)
shared_object = dirname + '/operations/secure_random/secure_random_module_tf_' + tf.__version__ + '.so'

try:
    secure_random_module = tf.load_op_library(shared_object)
except NotFoundError:
    logging.warning(
        "Falling back to insecure randomness since required custom op could not be "
        "found for the installed version of TensorFlow (" + tf.__version__ + "). "
        "Fix this by compiling custom ops.")
    secure_random_module = None


def supports_secure_randomness():
    return secure_random_module is not None


def supports_seeded_randomness():
    return secure_random_module is not None


def seeded_random_uniform(shape, minval=0, maxval=None, dtype=tf.int32, seed=None, name=None):
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

    return secure_random_module.secure_seeded_random_uniform(shape, seed, minval, maxval, name=name)


def random_uniform(shape, minval=0, maxval=None, dtype=tf.int32, name=None):
    """
    Returns cryptographically strong random numbers.

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

    minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
    maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")

    return secure_random_module.secure_random_uniform(shape, minval, maxval, name=name)


def seed():
    return secure_random_module.secure_seed()
