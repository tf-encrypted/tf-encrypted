import numpy as np
import math
from ..protocol import get_protocol

# TODO
# split backward function in compute_gradient and compute_backpropagated_error?


class Layer(object):

    @property
    def prot(self):
        return get_protocol()
