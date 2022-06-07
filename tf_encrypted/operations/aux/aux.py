import logging
import os

import tensorflow as tf
from tensorflow.python.framework.errors import NotFoundError

import tf_encrypted as tfe

logger = logging.getLogger("tf_encrypted")

SO_PATH = "{dn}/operations/aux/aux_module_tf_{tfv}.so"


def _try_load_aux_module():
    """
    Attempt to load and return aux module; returns None if failed.
    """
    so_file = SO_PATH.format(dn=os.path.dirname(tfe.__file__), tfv=tf.__version__)
    if not os.path.exists(so_file):
        logger.warning(
            (
                "Could not find aux module for the installed version of TensorFlow. "
                "Fix this by compiling custom ops. Missing file was '%s'"
            ),
            so_file,
        )
        return None

    try:
        module = tf.load_op_library(so_file)
        logger.info("aux module loaded: {}".format(module))
        return module

    except NotFoundError as ex:
        logger.warning(
            (
                "Could not find aux module for the installed version of TensorFlow. "
                "Fix this by compiling custom ops. Missing file was '%s', "
                'error was "%s".'
            ),
            so_file,
            ex,
        )

    except Exception as ex:  # pylint: disable=broad-except
        logger.error(
            (
                "Falling back to slow impl since an error occurred "
                'loading the required custom op: "%s".'
            ),
            ex,
        )

    return None


aux_module = _try_load_aux_module()


def bit_gather(x, start, stride):
    return aux_module.bit_gather(x, start=start, stride=stride)


def bit_split_and_gather(x, stride):
    return aux_module.bit_split_and_gather(x, stride=stride)


def bit_reverse(x: tf.Tensor):
    return aux_module.bit_reverse(x)


def xor_indices(x: tf.Tensor):
    return aux_module.xor_indices(x)
