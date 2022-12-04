import logging
import os
import tensorflow as tf
import numpy as np
import tf_encrypted as tfe
from tensorflow.python.framework.errors import NotFoundError

SO_PATH = "{dn}/operations/tf_i128/tf_i128_module_tf_{tfv}.so"
logger = logging.getLogger('tf_encrypted')

def _try_load_tf_i128_module():
    """
    Attempt to load and return tf_i128 module; returns None if failed.
    """
    so_file = SO_PATH.format(dn=os.path.dirname(tfe.__file__), tfv=tf.__version__)
    if not os.path.exists(so_file):
        logger.warning(
            (
                "Falling back to insecure randomness since the required custom op "
                "could not be found for the installed version of TensorFlow. Fix "
                "this by compiling custom ops. Missing file was '%s'"
            ),
            so_file,
        )
        return None

    try:
        module = tf.load_op_library(so_file)
        logger.info("tf_i128 module loaded: {}".format(module))
        return module

    except NotFoundError as ex:
        logger.warning(
            (
                "Falling back to insecure randomness since the required custom op "
                "could not be found for the installed version of TensorFlow. Fix "
                "this by compiling custom ops. "
                "Missing file was '%s', error was \"%s\"."
            ),
            so_file,
            ex,
        )

    except Exception as ex:  # pylint: disable=broad-except
        logger.error(
            (
                "Falling back to insecure randomness since an error occurred "
                'loading the required custom op: "%s".'
            ),
            ex,
        )

    return None


tf_i128 = _try_load_tf_i128_module()


def encode(A: np.ndarray):
    """
    A: (N, D) 2D array
    """
    assert isinstance(A, np.ndarray)
    # n_dims = len(A.shape)
    # assert n_dims >= 1 and n_dims <= 2

    shape = A.shape
    A = A.reshape([-1])
    A = A.astype(object) # cast to object to hold enough precision
    A = np.expand_dims(A, axis=-1)
    A = np.concatenate((A, np.zeros(A.shape, dtype=object)), axis=-1)

    def _encode(tupl):
        integer = int(tupl[0]) & ((1 << 128) - 1)
        return np.frombuffer(integer.to_bytes(16, 'little'), dtype=np.int64)
    A = np.apply_along_axis(_encode, -1, A)
    A = A.reshape(shape+(2,))
    return A

def decode(A: np.ndarray, scale: int):
    """
    A: (N, D, 2) 3D tensor
    scale: positive value
    """
    assert isinstance(A, np.ndarray)
    n_dims = len(A.shape)
    assert n_dims >= 1 and A.shape[-1] is 2
    assert scale > 0
    if isinstance (scale, float):
        print("WARN: (tf_i128.decode) the scale should be an int, but got {}".format(scale))
    scale = int(scale)
    def _decode_i128(two):
        return (int(two[0].astype(np.uint64)) + (int(two[1].astype(np.int64)) << 64)) / scale
    return np.apply_along_axis(_decode_i128, n_dims - 1, A).astype(np.float64)

def __is_tf_tensor(x):
    return isinstance(x, (tf.Tensor, tf.Variable))

def __is_i128_tensor(x):
    if not __is_tf_tensor(x): return False
    n_dims = len(x.shape)
    return n_dims > 0 and x.shape[-1] == 2

def __is_scalar_tensor(x):
    if not __is_i128_tensor(x): return False
    return np.prod(x.shape[:-1]) == 1

def reduce_sum(x: tf.Tensor, axis=None, keepdims=False):
    if axis is None:
        axis = -1
    assert __is_tf_tensor(x)
    assert axis < len(x.shape) - 1
    assert x.shape[-1] == 2

    def shape_after_reduce(shape, axis, keepdims):
        if axis < 0:
            return [1] * len(shape) if keepdims else []
        if keepdims:
            shape[axis] = 1
        else:
            shape = shape[:axis] + shape[axis+1:]
        return shape

    t = tf_i128.i128_reduce_sum(x, axis=axis, keepdims=bool(keepdims))
    reduced_shape = shape_after_reduce(x.shape.as_list()[:-1], axis, bool(keepdims))
    reduced_shape.append(2)
    t.set_shape(reduced_shape)
    return t

def to_i128(x: tf.Tensor, scale: int=1):
    assert(scale > 0)
    assert(__is_tf_tensor(x))
    x = tf.cast(x, tf.float64)
    if isinstance (scale, float):
        print("WARN: (tf_i128.to_i128) the scale should be an int, but got {}".format(scale))
    scale = int(scale)

    return tf_i128.to_i128(x, scale)

def from_i128(x: tf.Tensor, scale: int=1):
    assert(scale > 0)
    assert(__is_i128_tensor(x))
    if isinstance (scale, float):
        print("WARN: (tf_i128.from_i128) the scale should be an int, but got {}".format(scale))
    scale = int(scale)
    return tf_i128.from_i128(x, scale)

def mul(x: tf.Tensor, y: tf.Tensor):
    assert __is_i128_tensor(x) and __is_i128_tensor(y)
    return tf_i128.i128_mul(x, y)

def add(x: tf.Tensor, y: tf.Tensor):
    assert __is_i128_tensor(x) and __is_i128_tensor(y)
    return tf_i128.i128_add(x, y)

def sub(x: tf.Tensor, y: tf.Tensor):
    # assert __is_i128_tensor(x) and __is_i128_tensor(y)
    return tf_i128.i128_sub(x, y)

def matmul(x: tf.Tensor, y: tf.Tensor):
    assert __is_tf_tensor(x) and __is_tf_tensor(y)
    return tf_i128.i128_mat_mul(x, y)

def right_shift(x: tf.Tensor, shmt: int):
    assert __is_tf_tensor(x)
    return tf_i128.i128_right_shift(x, shmt)

def left_shift(x: tf.Tensor, shmt: int):
    assert __is_tf_tensor(x)
    return tf_i128.i128_left_shift(x, shmt)

def logic_right_shift(x: tf.Tensor, shmt: int):
    assert __is_tf_tensor(x)
    return tf_i128.i128_logic_right_shift(x, shmt)

def equal(x: tf.Tensor, y: tf.Tensor):
    assert __is_tf_tensor(x) and __is_tf_tensor(y)
    return tf_i128.i128_equal(x, y)

def negate(x: tf.Tensor):
    assert __is_i128_tensor(x)
    return tf_i128.i128_negate(x)

def i128_abs(x: tf.Tensor):
    assert __is_i128_tensor(x)
    return tf_i128.i128_abs(x)

def i128_bit_reverse(x: tf.Tensor):
    assert __is_i128_tensor(x)
    return tf_i128.i128_bit_reverse(x)

def i128_bit_gather(x: tf.Tensor, start, stride):
    assert __is_i128_tensor(x)
    return tf_i128.i128_bit_gather(x, start, stride)

def i128_bit_split_and_gather(x: tf.Tensor, stride):
    assert __is_i128_tensor(x)
    return tf_i128.i128_bit_split_and_gather(x, stride)

def i128_xor_indices(x: tf.Tensor):
    assert __is_i128_tensor(x)
    return tf_i128.i128_xor_indices(x)

