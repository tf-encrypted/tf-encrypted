

class OpError(Exception):
    '''Base tensorflow exception class
    implemented here:
    https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/framework/errors_impl.py#L32
    '''
    # TODO: need to add remainder of methods for class

    def __init__(self, node_def, op, message, error_code):
        """Creates a new `OpError` indicating that a particular op failed.
        Args:
        node_def: The `node_def_pb2.NodeDef` proto representing the op that
            failed, if known; otherwise None.
        op: The `ops.Operation` that failed, if known; otherwise None.
        message: The message string describing the failure.
        error_code: The `error_codes_pb2.Code` describing the error.
        """
        ...


class ResourceExhaustedError(OpError):
    # TODO: need to add remainder of methods for class
    ...
