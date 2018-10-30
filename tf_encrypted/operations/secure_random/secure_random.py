import tensorflow as tf
import tf_encrypted as tfe
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os

dirname = os.path.dirname(tfe.__file__)
shared_object = dirname + '/operations/secure_random/secure_random_module.so'


def secure_random(shape, minval=0, maxval=None, dtype=tf.int32, seed=None, name=None):
    # TODO docs

    secure_random_module = tf.load_op_library(shared_object)

    dtype = dtypes.as_dtype(dtype)
    if dtype not in (dtypes.int32, dtypes.int64):
        raise ValueError("Invalid dtype %r" % dtype)

    if maxval is None:
        raise ValueError("Must specify maxval for integer dtype %r" % dtype)

    if seed is None:
        # TODO random seed if not specified?
        seed = [87654321, 87654321, 87654321, 87654321, 87654321, 87654321, 87654321, 87654321]
    elif len(seed) != 8:
        raise ValueError("Seed length must 8")

    minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
    maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")

    return secure_random_module.secure_random(shape, seed, minval, maxval, name=name)
