from __future__ import absolute_import

import tensorflow as tf

_current_prot = None

def set_protocol(prot):
    global _current_prot
    _current_prot = prot

def get_protocol():
    return _current_prot

class Protocol(object):

    def __enter__(self):
        set_protocol(self)
        return self

    def __exit__(self, type, value, traceback):
        set_protocol(None)

# from unencrypted_native import UnencryptedNative
# from unencrypted_fixedpoint import UnencryptedFixedpoint
from .pond import Pond
from .securenn import SecureNN
# from secureml import SecureML